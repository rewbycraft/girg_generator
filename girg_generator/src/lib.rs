use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use clap::{ArgEnum, Parser, ValueHint};
use tracing::{debug, info};

use crate::parquet_edges::ParquetEdgeWriter;
use generator_common::params::VecSeeds;

#[cfg(feature = "benchmark")]
pub mod benchmark;
pub mod parquet_edges;
pub mod pbar;
#[cfg(test)]
pub mod tests;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
pub enum GeneratorMode {
    CPU,
    GPU,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
pub enum RandomMode {
    PreGenerate,
    OnDemand,
}

/// GIRG Generator
#[derive(Parser, Debug)]
pub struct Args {
    /// What generator to use
    #[clap(short, long, arg_enum)]
    pub generator: GeneratorMode,
    /// How to use the randomness
    #[clap(long, arg_enum)]
    pub random_mode: RandomMode,
    /// Number of worker threads
    #[clap(short, long, default_value_t = 1)]
    pub workers: usize,
    /// Number of vertices
    #[clap(long, default_value_t = 1000)]
    pub tile_size: u64,
    /// Number of vertices
    #[clap(short, long, default_value_t = 1_000_000)]
    pub vertices: u64,
    /// Alpha value of the probability function
    #[clap(short, long, default_value_t = 1.5)]
    pub alpha: f32,
    /// Alpha value of the pareto distribution
    #[clap(short, long, default_value_t = 1.5)]
    pub beta: f32,
    /// x_min value of the pareto distribution
    #[clap(short, long, default_value_t = 1.0)]
    pub x_min: f32,
    /// Number of spatial dimensions
    #[clap(short, long, default_value_t = 2)]
    pub dimensions: usize,
    /// Number of shards in use.
    #[clap(long, default_value_t = 1)]
    pub shard_count: usize,
    /// Index of this shard. [0,shard_count)
    #[clap(long, default_value_t = 0)]
    pub shard_index: usize,
    /// NVidia Device index
    #[clap(long, default_value_t = 0)]
    pub device: u32,
    /// Override the amount of blocks used by the gpu.
    #[clap(long)]
    pub blocks: Option<u32>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees_distribution to
    pub output_degrees_distribution: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (csv format: node_id, degree)
    pub output_degrees_csv: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (plain text: degree\n)
    pub output_degrees_txt: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write edges to (csv: i, j)
    pub output_edges_csv: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write edges to (parquet: i, j) (recommended due to compression)
    pub output_edges_parquet: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write weights to (plain text, one weight per line)
    pub output_weights: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write position to (csv: one column per dimension)
    pub output_positions: Option<PathBuf>,
    /// Seed values
    #[clap(long, short)]
    pub seeds: Option<Vec<u64>>,
    /// Size of the buffer used to hold edges before processing. Effectively edge batch size.
    #[clap(long, default_value_t = 1024)]
    pub edgebuffer_size: u64,
}

impl Args {
    pub fn get_pareto(&self) -> generator_common::random::ParetoDistribution {
        generator_common::random::ParetoDistribution::new(self.x_min, self.beta)
    }

    pub fn get_params(&self) -> generator_common::params::GenerationParameters<VecSeeds> {
        match self.seeds.as_ref() {
            None => generator_common::params::GenerationParameters::new(
                self.dimensions,
                self.get_pareto(),
                self.alpha,
                self.vertices,
                self.tile_size,
                self.edgebuffer_size,
                self.random_mode == RandomMode::PreGenerate,
                self.blocks,
            ),
            Some(s) => generator_common::params::GenerationParameters::from_seeds(
                self.dimensions,
                self.get_pareto(),
                self.alpha,
                self.vertices,
                s.as_slice(),
                self.tile_size,
                self.edgebuffer_size,
                self.random_mode == RandomMode::PreGenerate,
                self.blocks,
            ),
        }
    }
}

pub fn run_app(app: Args, ctx: Option<cust::context::Context>) -> anyhow::Result<()> {
    info!("Get params...");
    let params = app.get_params();

    info!("Params:\n{:#?}", params);

    if let Some(p) = app.output_weights {
        info!("Writing weights file...");
        let mut f = File::create(&p).expect("Unable to create file");
        for i in params.compute_weights() {
            writeln!(f, "{}", i).unwrap();
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
                if j < i.len() - 1 {
                    write!(f, ",").unwrap();
                }
            }
            writeln!(f).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    let (tile_sender, tile_receiver) = crossbeam_channel::bounded(5);
    let (edge_sender, edge_receiver) = crossbeam_channel::bounded(100);
    let (finish_sender, finish_receiver) = crossbeam_channel::bounded(10000);

    pbar::create_progress_bar(params.num_tiles());

    let mut handles = match app.generator {
        GeneratorMode::GPU => {
            generator_common::threads::start_workers::<generator_gpu::GPUGenerator>(
                &ctx,
                app.workers,
                edge_sender,
                finish_sender,
                tile_receiver,
                &params,
            )
        }
        GeneratorMode::CPU => {
            generator_common::threads::start_workers::<generator_cpu::CPUGenerator>(
                &ctx,
                app.workers,
                edge_sender,
                finish_sender,
                tile_receiver,
                &params,
            )
        }
    };
    handles.push(generator_common::threads::start_generate_tiles_thread(
        tile_sender,
        &params,
        app.shard_index,
        app.shard_count,
    ));

    let mut degree_counters: Vec<usize> = Vec::new();
    degree_counters.resize(app.vertices as usize, 0usize);

    handles.push(std::thread::spawn(move || {
        pbar::increment_progress(0);
        for b in finish_receiver {
            debug!("Finished block {:?}.", b);
            pbar::increment_progress(1);
        }
    }));

    {
        info!("Receiving edges...");
        let mut edge_counter = 0u128;

        let mut csv_wtr = app.output_edges_csv.map(|p| {
            let mut wtr = csv::Writer::from_path(&p).unwrap();
            wtr.write_record(&["edge_i", "edge_j"]).unwrap();
            wtr
        });

        let mut parquet_wtr = app.output_edges_parquet.map(ParquetEdgeWriter::new);

        for edge_tile in edge_receiver {
            if let Some(wtr) = parquet_wtr.as_mut() {
                wtr.write_vec(&edge_tile);
            }

            for (i, j) in edge_tile {
                edge_counter += 1;
                *degree_counters.get_mut(i as usize).unwrap() += 1;
                if let Some(wtr) = csv_wtr.as_mut() {
                    wtr.write_record(&[format!("{}", i), format!("{}", j)])
                        .unwrap();
                }
            }
        }

        if let Some(wtr) = csv_wtr.as_mut() {
            wtr.flush().unwrap();
        }

        if let Some(wtr) = parquet_wtr.as_mut() {
            wtr.close();
        }

        info!("All edges received! ({} edges)", edge_counter);
    }

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
            wtr.write_record(&[format!("{}", i), format!("{}", *j)])
                .unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_degrees_txt {
        info!("Writing degree txt...");
        let mut f = File::create(&p).expect("Unable to create file");
        for i in degree_counters.iter() {
            writeln!(f, "{}", i).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_degrees_distribution {
        info!("Writing degree distribution csv...");
        let mut wtr = csv::Writer::from_path(&p).unwrap();
        wtr.write_record(&["x", "number of nodes with degree > x / number of nodes"])
            .unwrap();
        for x in 0..=degree_counters.len() {
            let s: f64 = degree_counters.iter().filter(|&d| *d > x as usize).count() as f64;
            let v = s / (degree_counters.len() as f64);
            let x = x as f64;
            wtr.write_record(&[format!("{}", x), format!("{}", v)])
                .unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let app: Args = Args::parse();
    pbar::setup_logging(None);

    if app.shard_index >= app.shard_count {
        panic!("Shard index must be less than shard count.");
    }

    if app.dimensions < 1 {
        panic!("Number of dimensions must be at least 1.");
    }

    info!("Running using the {:?} generator!", app.generator);

    let ctx: Option<cust::context::Context> = if app.generator == GeneratorMode::GPU {
        info!("CUDA init...");
        cust::init(cust::CudaFlags::empty())?;
        info!("CUDA device get...");
        let device = cust::device::Device::get_device(app.device)?;

        info!(
            "Device: {} ({} MB)",
            device.name()?,
            device.total_memory()? / 1_000_000
        );

        Some(cust::context::Context::create_and_push(
            cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO,
            device,
        )?)
    } else {
        None
    };

    run_app(app, ctx)
}
