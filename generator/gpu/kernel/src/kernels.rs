use cuda_std::prelude::*;
use generator_core::algorithm::generate_edge;
use generator_core::params::{GenerationParameters, RawSeeds};
use generator_core::MAX_DIMS;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn generator_kernel(
    ts: *mut crate::state::gpu::GPUThreadState,
    params: &GenerationParameters<RawSeeds>,
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

    let (start, end) = params.pos_to_tile(ts.get_x(), ts.get_y());

    let mut i = ts.get_x();
    let mut j = ts.get_y();

    loop {
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

        if generate_edge(i, j, w(i), w(j), ps_i, ps_j, params) {
            if !ts.can_add_edge() {
                // No more space in buffer, abort!
                ts.set_done(false);
                break;
            }
            ts.add_edge(i, j);
        }

        // Increment i,j
        i += 1;
        if i >= params.v.min(end.0) {
            i = start.0;
            j += 1;
        }

        ts.set_x(i);
        ts.set_y(j);

        if j >= params.v.min(end.1) {
            // We've past the last node.

            ts.set_done(true);
            ts.set_x(end.0 - 1);
            ts.set_y(end.1 - 1);
            break;
        }
    }
}
