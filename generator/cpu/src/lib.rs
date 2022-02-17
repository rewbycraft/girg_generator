use crossbeam_channel::{Receiver, Sender};
use generator_common::algorithm::generate_edge;
use generator_common::params::ext::GenerationParametersExt;
use generator_common::params::{GenerationParameters, VecSeeds};
use tracing::{debug, info, instrument, warn};

#[inline]
pub fn worker_function<F: FnMut(u64, u64)>(
    start: (u64, u64),
    end: (u64, u64),
    params: &GenerationParameters<VecSeeds>,
    mut cb: F,
) {
    let mut i = start.0;
    let mut j = start.1;

    // Pre-calculate the params for j.
    let ws: Option<Vec<f32>> = if params.pregenerate_numbers {
        Some(params.compute_weights())
    } else {
        None
    };
    let ps: Option<Vec<Vec<f32>>> = if params.pregenerate_numbers {
        Some(params.compute_positions())
    } else {
        None
    };
    let mut p_i_prime = Vec::new();
    let mut p_j_prime = Vec::new();
    p_i_prime.resize(params.num_dimensions(), 0.0f32);
    p_j_prime.resize(params.num_dimensions(), 0.0f32);

    loop {
        let w_i = ws
            .as_ref()
            .map(|w| *w.get(i as usize).unwrap())
            .unwrap_or_else(|| params.compute_weight(i));
        let w_j = ws
            .as_ref()
            .map(|w| *w.get(j as usize).unwrap())
            .unwrap_or_else(|| params.compute_weight(j));
        let p_i = ps
            .as_ref()
            .map(|p| p.get(i as usize).unwrap())
            .unwrap_or_else(|| {
                params.fill_dims(i, &mut p_i_prime);
                &p_i_prime
            });
        let p_j = ps
            .as_ref()
            .map(|p| p.get(j as usize).unwrap())
            .unwrap_or_else(|| {
                params.fill_dims(j, &mut p_j_prime);
                &p_j_prime
            });

        if generate_edge(i, j, w_i, w_j, p_i, p_j, params) {
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

pub struct CPUGenerator {}

impl generator_common::generator::GraphGenerator for CPUGenerator {
    type ConstructArgument = ();

    fn new(_: Self::ConstructArgument) -> anyhow::Result<Self> {
        Ok(Self {})
    }

    #[instrument(skip_all)]
    fn generate(
        &self,
        sender: Sender<Vec<(u64, u64)>>,
        finisher: Sender<((u64, u64), (u64, u64))>,
        receiver: Receiver<((u64, u64), (u64, u64))>,
        params: &GenerationParameters<VecSeeds>,
    ) -> anyhow::Result<()> {
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

pub fn worker(
    sender: Sender<Vec<(u64, u64)>>,
    start: (u64, u64),
    end: (u64, u64),
    params: &GenerationParameters<VecSeeds>,
) {
    let mut pair_queue = Vec::new();
    pair_queue.resize(params.edgebuffer_size as usize, (0, 0));
    let mut pair_queue_index = 0usize;
    let mut pair_queue_sends = 0usize;

    info!("Job: {:?} -> {:?}", start, end);
    crate::worker_function(start, end, params, |i, j| {
        pair_queue[pair_queue_index] = (i, j);
        pair_queue_index += 1;

        if pair_queue_index >= pair_queue.len() {
            let v = pair_queue.clone();

            debug!("Sending {} pairs.", v.len());

            sender.send(v).unwrap();
            pair_queue_index = 0;
            pair_queue_sends += 1;
        }
    });

    if pair_queue_index > 0 {
        let v = Vec::from(&pair_queue[0..pair_queue_index]);
        debug!("Sending {} pairs.", v.len());
        sender.send(v).unwrap();
        pair_queue_sends += 1;
    }

    if pair_queue_sends > 1 {
        warn!("Edge buffer likely too small. Had to send more than one for this job. Consider increasing the edgebuffer size to {}.", params.edgebuffer_size as usize * pair_queue_sends);
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
