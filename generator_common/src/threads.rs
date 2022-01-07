use std::thread::JoinHandle;
use tracing::{info, instrument};
use crossbeam_channel::{Sender, Receiver};
use crate::params::{GenerationParameters, VecSeeds};
use cust::context::{Context, CurrentContext};
use crate::generator::GraphGenerator;

pub fn start_generate_tiles_thread(sender: Sender<((u64, u64), (u64, u64))>, params: &GenerationParameters<VecSeeds>, shard_index: usize, shard_count: usize) -> JoinHandle<()> {
    let params = params.clone();
    std::thread::spawn(move || {
        generate_tiles(sender, &params, shard_index, shard_count)
    })
}

pub fn generate_tiles(sender: Sender<((u64, u64), (u64, u64))>, params: &GenerationParameters<VecSeeds>, shard_index: usize, shard_count: usize) {
    info!("Emitting tiles...");

    for tile in params.tiles().skip(shard_index).step_by(shard_count) {
        sender.send(tile).unwrap();
    }

    info!("Tiles are generated!");
}

pub fn start_workers<T: GraphGenerator>(ctx: &Option<Context>, num_workers: usize, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &crate::GenerationParameters<VecSeeds>) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();

    for i in 0u64..(num_workers as u64) {
        let sender = sender.clone();
        let receiver = receiver.clone();
        let finisher = finisher.clone();
        let params = params.clone();
        let unowned = ctx.as_ref().map(|c| c.get_unowned());
        handles.push(std::thread::spawn(move || {
            if let Some(c) = unowned.as_ref() {
                CurrentContext::set_current(c).unwrap();
            }
            worker_thread::<T>(i, sender, finisher, receiver, &params);
        }));
    }

    drop(sender);
    drop(receiver);
    drop(finisher);

    handles
}

#[instrument(skip_all, fields(tid = _thread_id))]
pub fn worker_thread<T: GraphGenerator>(_thread_id: u64, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &GenerationParameters<VecSeeds>) {
    info!("Running!");
    let generator = T::new().unwrap();
    generator.generate(sender, finisher, receiver, params).unwrap();
    info!("Thread exit.");
}
