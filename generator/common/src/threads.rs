use crate::generator::{EdgeSender, GraphGenerator};
use crate::tiles::Tile;
use crossbeam_channel::{Receiver, Sender};
use std::thread::JoinHandle;
use tracing::{info, instrument};
use crate::params::CPUGenerationParameters;

pub fn start_generate_tiles_thread(
    sender: Sender<Tile>,
    params: &CPUGenerationParameters,
) -> JoinHandle<()> {
    let params = params.clone();
    std::thread::spawn(move || generate_tiles(sender, &params))
}

pub fn generate_tiles(sender: Sender<Tile>, params: &CPUGenerationParameters) {
    info!("Emitting tiles...");

    for tile in params.tiles() {
        sender.send(tile).unwrap();
    }

    info!("Tiles are generated!");
}

pub fn start_workers<T: GraphGenerator>(
    construct_arg: T::ConstructArgument,
    num_workers: usize,
    sender: EdgeSender,
    finisher: Sender<Tile>,
    receiver: Receiver<Tile>,
    params: &CPUGenerationParameters,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();

    for i in 0u64..(num_workers as u64) {
        let sender = sender.clone();
        let receiver = receiver.clone();
        let finisher = finisher.clone();
        let params = params.clone();
        let construct_arg = construct_arg.clone();
        handles.push(std::thread::spawn(move || {
            worker_thread::<T>(i, construct_arg, sender, finisher, receiver, &params);
        }));
    }

    drop(sender);
    drop(receiver);
    drop(finisher);

    handles
}

#[instrument(skip_all, fields(tid = _thread_id))]
pub fn worker_thread<T: GraphGenerator>(
    _thread_id: u64,
    construct_arg: T::ConstructArgument,
    sender: EdgeSender,
    finisher: Sender<Tile>,
    receiver: Receiver<Tile>,
    params: &CPUGenerationParameters,
) {
    info!("Running!");
    let generator = T::new(construct_arg).unwrap();
    generator
        .generate(sender, finisher, receiver, params)
        .unwrap();
    info!("Thread exit.");
}
