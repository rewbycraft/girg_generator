use crossbeam_channel::{Receiver, Sender};
use tracing::{info, debug, instrument};

use generator_common::params::GenerationParameters;

// #[inline]
// pub fn worker_function_compute<F: FnMut(u64, u64)>(start: (u64, u64), end: (u64, u64), params: &GenerationParameters, mut cb: F) {
//     let mut i = start.0;
//     let mut j = start.1;
//
//     // Pre-calculate the params for j.
//     let mut w_j = random::uniform_to_pareto(random::random_property(j, params.seeds[0]), &params.pareto);
//     let mut p0_j = random::random_property(j, params.seeds[2]);
//     let mut p1_j = random::random_property(j, params.seeds[3]);
//
//     loop {
//         let w_i = random::uniform_to_pareto(random::random_property(i, params.seeds[0]), &params.pareto);
//         let p0_i = random::random_property(i, params.seeds[2]);
//         let p1_i = random::random_property(i, params.seeds[3]);
//
//         if generator_common::generate_edge(i, j, w_i, p0_i, p1_i, w_j, p0_j, p1_j, params) {
//             cb(i, j)
//         }
//
//         // Increment i,j
//         i += 1;
//         if i >= params.v.min(end.0) {
//             i = start.0;
//             j += 1;
//
//             // Re-calculate the params for j.
//             w_j = random::uniform_to_pareto(random::random_property(j, params.seeds[0]), &params.pareto);
//             p0_j = random::random_property(j, params.seeds[2]);
//             p1_j = random::random_property(j, params.seeds[3]);
//         }
//         if j >= params.v.min(end.1) {
//             // We've past the last node.
//             break;
//         }
//         if i >= params.v.min(end.0) && j >= params.v.min(end.1) {
//             // We're done.
//             break;
//         }
//     }
// }

#[inline]
pub fn worker_function_pregen<F: FnMut(u64, u64)>(start: (u64, u64), end: (u64, u64), params: &GenerationParameters, mut cb: F) {
    let mut i = start.0;
    let mut j = start.1;

    // Pre-calculate the params for j.
    let ws: Vec<f32> = params.compute_weights();
    let ps = params.compute_positions();

    loop {
        if generator_common::generate_edge(
            i,
            j,
            *ws.get(i as usize).unwrap(),
            *ws.get(j as usize).unwrap(),
            *ps.get(i as usize).unwrap(),
            *ps.get(j as usize).unwrap(),
            params,
        )
        {
            cb(i, j)
        }

        // Increment i,j
        i += 1;
        if i >= params.v.min(end.0) {
            i = start.0;
            j += 1;
        }
        if j >= params.v.min(end.1) {
            // We've past the last node.
            break;
        }
        if i >= end.0 && j >= end.1 {
            // We're done.
            break;
        }
    }
}

pub struct CPUGenerator {
}

impl generator_common::generator::GraphGenerator for CPUGenerator {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {})
    }

    #[instrument(skip_all)]
    fn generate(&self, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &GenerationParameters) -> anyhow::Result<()> {
        info!("Running!");
        for (start, end) in receiver {
            worker(sender.clone(), start, end, params);
            finisher.send((start, end)).unwrap();
        }
        drop(sender);
        drop(finisher);
        info!("Thread exit.");
        Ok(())
    }
}

pub fn worker(sender: Sender<Vec<(u64, u64)>>, start: (u64, u64), end: (u64, u64), params: &GenerationParameters) {
    let mut pair_queue = [(0, 0); 40960];
    let mut pair_queue_index = 0usize;

    info!("Job: {:?} -> {:?}", start, end);
    crate::worker_function_pregen(start, end, params, |i, j| {
        pair_queue[pair_queue_index] = (i, j);
        pair_queue_index += 1;

        if pair_queue_index >= pair_queue.len() {
            let v = Vec::from(pair_queue);

            debug!("Sending {} pairs.", v.len());

            sender.send(v).unwrap();
            pair_queue_index = 0;
        }
    });

    if pair_queue_index > 0 {
        let v = Vec::from(&pair_queue[0..pair_queue_index]);
        debug!("Sending {} pairs.", v.len());
        sender.send(v).unwrap();
    }

    info!("Job done!");
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
