use crossbeam_channel::{Receiver, Sender};
use cust::error::CudaResult;
use cust::memory::{DeviceBox, GpuBuffer};
use cust::prelude::*;
use tracing::{debug, info, instrument, warn};
use generator_common::generator::GraphGenerator;

use generator_common::params::{GenerationParameters, VecSeeds};

static PTX: &str = include_str!(env!("KERNEL_PTX_PATH"));

pub struct GPUGenerator {
    module: Module,
}

impl GPUGenerator {

    fn launch_run(&self,
                  cpu_state: &mut gpu_kernel::state::cpu::CPUThreadState,
                  grid_size: u32,
                  block_size: u32,
                  stream: &Stream,
                  params: &GenerationParameters<VecSeeds>,
                  variables_d: &mut DeviceBuffer<f32>,
    ) -> CudaResult<()> {
        info!("Starting a run...");
        let kernel_function = self.module.get_function("generator_kernel")?;

        // Queue up the commands to the GPU.
        unsafe {
            let seeds_buffer = params.seeds.get_dbuffer_async(stream)?;
            let params_raw = GenerationParameters::from_vecseeds_and_device_ptr(params, seeds_buffer.as_device_ptr());
            let mut params_d = DeviceBox::new(&params_raw)?;
            cpu_state.copy_to_device_async(stream).unwrap();

            let mut gpu_state = cpu_state.create_gpu_state().unwrap();

            launch!(
                // slices are passed as two parameters, the pointer and the length.
                kernel_function<<<grid_size, block_size, 0, stream>>>(
                    gpu_state.as_device_ptr(),
                    params_d.as_device_ptr(),
                    variables_d.as_device_ptr(),
                    variables_d.len(),
                )
            ).unwrap();

            cpu_state.copy_from_device_async(stream).unwrap();
        }

        // Wait for the GPU to finish the job.
        stream.synchronize().unwrap();

        info!("Run complete!");

        Ok(())
    }
}

pub fn suggested_launch_configuration() -> CudaResult<(u32, u32)> {
    let module = Module::from_str(PTX)?;
    let kernel_function = module.get_function("generator_kernel")?;
    let sg = kernel_function.suggested_launch_configuration(0, 0.into())?;
    Ok(sg)
}

impl GraphGenerator for GPUGenerator {
    fn new() -> anyhow::Result<Self> {
        let module = Module::from_str(PTX)?;
        Ok(Self {
            module,
        })
    }

    #[instrument(skip_all)]
    fn generate(&self, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &GenerationParameters<VecSeeds>) -> anyhow::Result<()> {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let variables = params.compute_interleaved_variables();
        let mut variables_d = variables.as_slice().as_dbuf()?;

        let kernel_function = self.module.get_function("generator_kernel")?;
        let (grid_size, block_size) = kernel_function.suggested_launch_configuration(0, 0.into())?;

        let grid_size = grid_size.min((params.num_tiles() as u32 + block_size - 1) / block_size);

        let grid_size = params.gpu_blocks.unwrap_or(grid_size);

        let num_threads = (grid_size * block_size) as usize;

        info!(
            "Using {} blocks and {} threads per block for a total of {} GPU threads.",
            grid_size, block_size, num_threads
        );

        let mut cpu_state = gpu_kernel::state::cpu::CPUThreadState::new(params.edgebuffer_size, num_threads as u64)?;

        //Mark all threads as done. This way they'll all get a new tile.
        cpu_state.done.fill(true);

        let mut edge_queue: Vec<(u64, u64)> = Vec::new();

        let mut avg_overfill_sum = 0.0f64;
        let mut avg_overfill_count = 0usize;
        info!("Beginning generation loop...");
        loop {
            info!("Starting round...");
            let mut alloc_counter = 0usize;
            let mut block_counter = 0usize;
            for tid in 0..num_threads {
                // Check if this thread is ready for a new tile.
                if cpu_state.done[tid] {
                    // This thread is done, try and allocate a new tile.
                    if let Ok(((start_left, start_right), (end_left, end_right))) = receiver.recv() {
                        debug!("Allocated tile ({}, {}) -> ({}, {}) to GPU thread {}.", start_left, start_right, end_left, end_right, tid);
                        alloc_counter += 1;
                        // New tile get! Set it in the state.
                        cpu_state.done[tid] = false;
                        cpu_state.current_x[tid] = start_left;
                        cpu_state.current_y[tid] = start_right;
                    }
                }

                if !cpu_state.done[tid] {
                    block_counter += 1;
                }
            }
            info!("Allocated {} blocks this round!", alloc_counter);

            // If they're all done even after filling with tiles, then we're out of tiles.
            if cpu_state.done.iter().all(|v| *v) {
                break;
            }

            let old_done = cpu_state.done.clone();

            // Run the current round
            self.launch_run(&mut cpu_state, grid_size, block_size, &stream, params, &mut variables_d)?;

            debug!("Debug: {:?}", cpu_state.debug);

            // Clean out the current edge queue.
            edge_queue.clear();
            for tid in 0..num_threads {
                // Read all the edges into the queue.
                edge_queue.extend(cpu_state.edges_iter(tid));
                // Remove them from the gpu side.
                cpu_state.edges_count[tid] = 0;
            }

            //Send the edges off.
            if edge_queue.len() > 0 {
                sender.send(edge_queue.clone()).unwrap();
            }

            // Send out notifications that a tile is done.
            for tid in 0..num_threads {
                if !old_done[tid] && cpu_state.done[tid] {
                    // We just finished the tile.
                    // Note that because of this it is *CRUCIAL* that the kernel leave the x,y set to the last position in the tile.
                    let p = params.pos_to_tile(cpu_state.current_x[tid], cpu_state.current_y[tid]);
                    finisher.send(p).unwrap();
                }
            }

            let avg_fill = (edge_queue.len() as f64) / (block_counter as f64);
            info!("Finished round having generated {} edges ({:.02} edges per thread).", edge_queue.len(), avg_fill);
            if avg_fill > (params.edgebuffer_size as f64) * 0.9 {
                avg_overfill_sum += avg_fill;
                avg_overfill_count += 1;
                let recommended_size = (avg_overfill_sum / (avg_overfill_count as f64)) * (avg_overfill_count as f64 + 1.0);
                warn!("Fill was over 90% of the buffer! Consider increasing the edge buffer size to {}.", recommended_size);
            }
        }

        info!("Done generating!");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
