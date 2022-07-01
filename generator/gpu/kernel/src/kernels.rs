//! Actual kernel function.

use cuda_std::prelude::*;
use generator_core::algorithm::generate_edge;
use generator_core::params::GenerationParameters;
use generator_core::MAX_DIMS;

/// Main kernel entrypoint for the GPU.
///
/// Must be invoked via the [`cust::launch`] macro.
///
/// # Arguments
/// * `ts` - GPU Resident thread state.
/// * `params` - GPU Resident instance of the [`GenerationParameters`].
/// * `variables` - Interleaved node variables. See [`generator_common::params::CPUGenerationParameters::compute_interleaved_variables()`].
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn generator_kernel(
    ts: *mut crate::state::gpu::GPUThreadState,
    params: GenerationParameters,
    variables: &[f32],
) {
    let ts = &mut *ts;
    if thread::index_1d() >= ts.num_threads as u32 {
        panic!("too many threads");
    }

    if ts.get_done() {
        // We're not needed anymore. Return.
        return;
    }

    let mut p_i_prime = [0.0f32; MAX_DIMS];
    let mut p_j_prime = [0.0f32; MAX_DIMS];

    let w = |i: u64| {
        if params.pregenerate_numbers {
            variables[(i as usize) * (params.num_dimensions() + 1)]
        } else {
            params.compute_weight(i)
        }
    };
    let ps = |i: u64| {
        &variables[(((i as usize) * (params.num_dimensions() + 1)) + 1)
            ..(((i as usize) * (params.num_dimensions() + 1)) + (params.num_dimensions() + 1))]
    };

    let tile = params.pos_to_tile(ts.get_x(), ts.get_y());

    let mut failed = false;
    for (i, j) in tile.into_iter().skip_to(ts.get_x(), ts.get_y()) {
        let ps_i: &[f32] = if params.pregenerate_numbers {
            ps(i)
        } else {
            params.fill_dims(i, &mut p_i_prime[0..params.num_dimensions()]);
            &p_i_prime[0..params.num_dimensions()]
        };
        let ps_j: &[f32] = if params.pregenerate_numbers {
            ps(j)
        } else {
            params.fill_dims(j, &mut p_j_prime[0..params.num_dimensions()]);
            &p_j_prime[0..params.num_dimensions()]
        };

        if generate_edge(i, j, w(i), w(j), ps_i, ps_j, &params) {
            if !ts.can_add_edge() {
                // No more space in buffer, abort!
                failed = true;
                break;
            }
            ts.add_edge(i, j);
        }

        ts.set_x(i);
        ts.set_y(j);
    }

    ts.set_done(!failed);
}
