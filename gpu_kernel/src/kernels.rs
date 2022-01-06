use cuda_std::prelude::*;

use generator_common::{compute_distance, compute_probability, generate_edge, random};
use generator_common::params::SeedEnum;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn generator_kernel(ts: *mut crate::state::gpu::GPUThreadState, params: &generator_common::params::GenerationParameters, variables: &[f32]) {
    let ts = &mut *ts;
    if thread::index_1d() >= ts.num_threads as u32 {
        panic!("too many threads");
    }

    if ts.get_done() {
        // We're not needed anymore. Return.
        return;
    }

    let w = |i: u64| { variables[(i as usize) * (params.num_dimensions() + 1)] };
    let p = |i: u64, d: usize| { variables[((i as usize) * (params.num_dimensions() + 1)) + (d + 1)] };
    let ps = |i: u64| {
        let mut a = [0f32; generator_common::params::DIMENSIONS];
        for d in 0..a.len() {
            a[d] = p(i, d);
        }
        a
    };

    let (start, end) = params.pos_to_tile(ts.get_x(), ts.get_y());

    let mut i = ts.get_x();
    let mut j = ts.get_y();

    loop {
        if generate_edge(i,
                         j,
                         w(i),
                         w(j),
                         ps(i),
                         ps(j),
                         params)
        {
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

