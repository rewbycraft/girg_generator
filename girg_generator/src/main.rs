use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use clap::{ArgEnum, Parser, ValueHint};
use tracing::{debug, info};

pub mod pbar;
#[cfg(test)]
pub mod tests;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
enum GeneratorMode {
    CPU,
    GPU,
}

/// GIRG Generator
#[derive(Parser, Debug)]
struct Args {
    /// What generator to use
    #[clap(short, long, arg_enum)]
    generator: GeneratorMode,
    /// Number of worker threads
    #[clap(short, long, default_value_t = 1)]
    workers: usize,
    /// Number of vertices
    #[clap(long, default_value_t = 1000)]
    tile_size: u64,
    /// Number of vertices
    #[clap(short, long, default_value_t = 1_000_000)]
    vertices: u64,
    /// Alpha value of the probability function
    #[clap(short, long, default_value_t = 1.5)]
    alpha: f32,
    /// Alpha value of the pareto distribution
    #[clap(short, long, default_value_t = 1.5)]
    beta: f32,
    /// x_min value of the pareto distribution
    #[clap(short, long, default_value_t = 1.0)]
    x_min: f32,
    /// Number of shards in use.
    #[clap(long, default_value_t = 1)]
    shard_count: usize,
    /// Index of this shard. [0,shard_count)
    #[clap(long, default_value_t = 0)]
    shard_index: usize,
    /// NVidia Device index
    #[clap(short, long, default_value_t = 0)]
    device: u32,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees_distribution to
    output_degrees_distribution: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (csv format: node_id, degree)
    output_degrees_csv: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (plain text: degree\n)
    output_degrees_txt: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write edges to (csv: i, j)
    output_edges: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write weights to (plain text, one weight per line)
    output_weights: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write position to (csv: one column per dimension)
    output_positions: Option<PathBuf>,
    /// Seed values
    #[clap(long, short)]
    seeds: Option<Vec<u64>>,
}

impl Args {
    pub fn get_pareto(&self) -> generator_common::random::ParetoDistribution {
        generator_common::random::ParetoDistribution::new(self.x_min, self.beta)
    }

    pub fn get_params(&self) -> generator_common::params::GenerationParameters {
        match self.seeds.as_ref() {
            None => generator_common::params::GenerationParameters::new(self.get_pareto(), self.alpha, self.vertices, self.tile_size),
            Some(s) => generator_common::params::GenerationParameters::from_seeds(self.get_pareto(), self.alpha, self.vertices, s.as_slice(), self.tile_size),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let app: Args = Args::parse();
    pbar::setup_logging();

    if app.shard_index >= app.shard_count {
        panic!("Shard index must be less than shard count.");
    }

    info!("Running using the {:?} generator!", app.generator);

    let ctx: Option<cust::context::Context> = if app.generator == GeneratorMode::GPU {
        info!("CUDA init...");
        cust::init(cust::CudaFlags::empty())?;
        info!("CUDA device get...");
        let device = cust::device::Device::get_device(app.device)?;

        info!("Device: {} ({} MB)", device.name()?, device.total_memory()? / 1_000_000);

        Some(cust::context::Context::create_and_push(cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO, device)?)
    } else {
        None
    };

    info!("Get params...");
    let params = app.get_params();

    info!("Params:\n{:#?}", params);

    if let Some(p) = app.output_weights {
        info!("Writing weights file...");
        let mut f = File::create(&p).expect("Unable to create file");
        for i in params.compute_weights() {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_positions {
        info!("Writing positions file...");
        let mut f = File::create(&p).expect("Unable to create file");
        for i in params.compute_positions() {
            for j in 0..i.len() {
                write!(f, "{}", i[j]).unwrap();
                if j < i.len()-1 {
                    write!(f, ",").unwrap();
                }
            }
            write!(f, "\n").unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }


    let (tile_sender, tile_receiver) = crossbeam_channel::bounded(5);
    let (edge_sender, edge_receiver) = crossbeam_channel::bounded(100);
    let (finish_sender, finish_receiver) = crossbeam_channel::bounded(10000);

    pbar::create_progress_bar(params.num_tiles());

    let mut handles = match app.generator {
        GeneratorMode::GPU => generator_common::threads::start_workers::<gpu_generator::GPUGenerator>(&ctx, app.workers, edge_sender, finish_sender, tile_receiver, &params),
        GeneratorMode::CPU => generator_common::threads::start_workers::<cpu_generator::CPUGenerator>(&ctx, app.workers, edge_sender, finish_sender, tile_receiver, &params),
    };
    handles.push(generator_common::threads::start_generate_tiles_thread(tile_sender, &params, app.shard_index, app.shard_count));

    let mut degree_counters: Vec<usize> = Vec::new();
    degree_counters.resize(app.vertices as usize, 0usize);

    handles.push(std::thread::spawn(move || {
        pbar::increment_progress(0);
        for b in finish_receiver {
            debug!("Finished block {:?}.", b);
            pbar::increment_progress(1);
        }
    }));

    info!("Receiving edges...");
    let mut edge_counter = 0u128;
    if let Some(p) = app.output_edges {
        let mut wtr = csv::Writer::from_path(&p).unwrap();
        wtr.write_record(&["edge_i", "edge_j"]).unwrap();

        for edge_tile in edge_receiver {
            for (i, j) in edge_tile {
                edge_counter += 1;
                wtr.write_record(&[format!("{}", i), format!("{}", j)]).unwrap();
                *degree_counters.get_mut(i as usize).unwrap() += 1;
            }
        }
        wtr.flush().unwrap();
    } else {
        for edge_tile in edge_receiver {
            for (i, _j) in edge_tile {
                edge_counter += 1;
                *degree_counters.get_mut(i as usize).unwrap() += 1;
            }
        }
    }
    info!("All edges received! ({} edges)", edge_counter);

    info!("Waiting for the threads to join...");
    while let Some(h) = handles.pop() {
        h.join().unwrap();
    }
    info!("Threads joined!");

    pbar::finish_progress_bar();

    //info!("Degrees: {:?}", degree_counters);

    if let Some(p) = app.output_degrees_csv {
        info!("Writing degree csv...");
        let mut wtr = csv::Writer::from_path(&p).unwrap();
        wtr.write_record(&["node_id", "degree"]).unwrap();
        for (i, j) in degree_counters.iter().enumerate() {
            wtr.write_record(&[format!("{}", i), format!("{}", *j)]).unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_degrees_txt {
        info!("Writing degree txt...");
        let mut f = File::create(&p).expect("Unable to create file");
        for i in degree_counters.iter() {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }


    if let Some(p) = app.output_degrees_distribution {
        info!("Writing degree distribution csv...");
        let mut wtr = csv::Writer::from_path(&p).unwrap();
        wtr.write_record(&["x", "number of nodes with degree > x / number of nodes"]).unwrap();
        for x in 0..=degree_counters.len() {
            let s: f64 = degree_counters.iter().filter(|&d| *d > x as usize).count() as f64;
            let v = s / (degree_counters.len() as f64);
            let x = x as f64;
            wtr.write_record(&[format!("{}", x), format!("{}", v)]).unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    Ok(())
}
