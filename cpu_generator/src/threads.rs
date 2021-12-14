use std::thread::JoinHandle;
use crossbeam_channel::{Receiver, Sender};
use tracing::{info, debug};

pub fn start_workers(num_workers: usize, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &crate::GenerationParameters) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();

    for i in 0u64..(num_workers as u64) {
        let sender = sender.clone();
        let receiver = receiver.clone();
        let finisher = finisher.clone();
        let params = params.clone();
        handles.push(std::thread::spawn(move || {
            worker_thread(i, sender, finisher, receiver, &params);
        }));
    }

    drop(sender);
    drop(receiver);
    drop(finisher);

    handles
}

pub fn start_generate_blocks_thread(sender: Sender<((u64, u64), (u64, u64))>, block_size: u64, params: &crate::GenerationParameters) -> JoinHandle<()> {
    let params = params.clone();
    std::thread::spawn(move || {
        generate_blocks(sender, block_size, &params)
    })
}

pub fn generate_blocks(sender: Sender<((u64, u64), (u64, u64))>, block_size: u64, params: &crate::GenerationParameters) {
    info!("Emitting blocks...");

    for block in params.blocks(block_size) {
        sender.send(block).unwrap();
    }

    info!("Blocks are generated!");
}

pub fn worker_thread(thread_id: u64, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &crate::GenerationParameters) {
    info!("thread[{}]: Running!", thread_id);
    for (start, end) in receiver {
        worker(thread_id, sender.clone(), start, end, params);
        finisher.send((start, end)).unwrap();
    }
    drop(sender);
    drop(finisher);
    info!("thread[{}]: Thread exit.", thread_id);
}

pub fn worker(thread_id: u64, sender: Sender<Vec<(u64, u64)>>, start: (u64, u64), end: (u64, u64), params: &crate::GenerationParameters) {
    let mut pair_queue = [(0, 0); 40960];
    let mut pair_queue_index = 0usize;

    info!("thread[{}]: Job: {:?} -> {:?}", thread_id, start, end);
    crate::worker_function_pregen(start, end, params, |i, j| {
        pair_queue[pair_queue_index] = (i, j);
        pair_queue_index += 1;

        if pair_queue_index >= pair_queue.len() {
            let v = Vec::from(pair_queue);

            debug!("thread[{}]: Sending {} pairs.", thread_id, v.len());

            sender.send(v).unwrap();
            pair_queue_index = 0;
        }
    });

    if pair_queue_index > 0 {
        let v = Vec::from(&pair_queue[0..pair_queue_index]);
        debug!("thread[{}]: Sending {} pairs.", thread_id, v.len());
        sender.send(v).unwrap();
    }

    info!("thread[{}]: Job done!", thread_id);
}
