use std::io::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use clap::{Parser, ValueHint};
use tracing::info;

pub mod pbar;

/// Simple program to greet a person
#[derive(Parser, Debug)]
struct Args {
    /// Number of vertices
    #[clap(short, long, default_value_t = 4)]
    workers: usize,
    /// Number of vertices
    #[clap(long, default_value_t = 5000)]
    block_size: u64,
    /// Number of vertices
    #[clap(short, long, default_value_t = 100000)]
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
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "degrees_distribution.csv")]
    /// File to write degrees_distribution to
    pub output_degrees_distribution: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "degrees.csv")]
    /// File to write degrees to
    pub output_degrees: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "degrees.txt")]
    /// File to write degrees to
    pub output_degrees_txt: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "edges.csv")]
    /// File to write edges to
    pub output_edges: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "weights.txt")]
    /// File to write degrees to
    pub output_weights: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "p0.txt")]
    /// File to write degrees to
    pub output_p0: PathBuf,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "p1.txt")]
    /// File to write degrees to
    pub output_p1: PathBuf,
}

impl Args {
    pub fn get_pareto(&self) -> cpu_generator::random::ParetoDistribution {
        cpu_generator::random::ParetoDistribution::new(self.x_min, self.beta)
    }

    pub fn get_params(&self) -> cpu_generator::GenerationParameters {
        cpu_generator::GenerationParameters::new(self.get_pareto(), self.alpha, self.vertices)
    }
}

fn main() {
    pbar::setup_logging();

    let app: Args = Args::parse();
    let params = app.get_params();

    info!("Params:\n{:#?}", params);

    info!("Writing weights file...");
    {
        let mut f = File::create(app.output_weights).expect("Unable to create file");
        for i in params.compute_weights() {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
    }
    info!("Done writing!");

    info!("Writing p0 file...");
    {
        let mut f = File::create(app.output_p0).expect("Unable to create file");
        for i in params.compute_positions(0) {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
    }
    info!("Done writing!");

    info!("Writing p1 file...");
    {
        let mut f = File::create(app.output_p1).expect("Unable to create file");
        for i in params.compute_positions(1) {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
    }
    info!("Done writing!");

    let (block_sender, block_receiver) = crossbeam_channel::bounded(5);
    let (edge_sender, edge_receiver) = crossbeam_channel::bounded(5);
    let (finish_sender, finish_receiver) = crossbeam_channel::bounded(5);

    pbar::create_progress_bar(params.num_blocks(app.block_size));

    let mut handles = cpu_generator::threads::start_workers(app.workers, edge_sender, finish_sender, block_receiver, &params);
    handles.push(cpu_generator::threads::start_generate_blocks_thread(block_sender, app.block_size, &params));

    let mut degree_counters: Vec<usize> = Vec::new();
    degree_counters.resize(app.vertices as usize, 0usize);

    handles.push(std::thread::spawn(move || {
        for _ in finish_receiver {
            pbar::increment_progress(1);
        }
    }));

    info!("Receiving edges...");
    {
        let mut wtr = csv::Writer::from_path(&app.output_edges).unwrap();
        wtr.write_record(&["edge_i", "edge_j"]).unwrap();

        for edge_block in edge_receiver {
            for (i, j) in edge_block {
                wtr.write_record(&[format!("{}", i), format!("{}", j)]).unwrap();
                *degree_counters.get_mut(i as usize).unwrap() += 1;
            }
        }
        wtr.flush().unwrap();
    }
    info!("All edges received!");

    info!("Waiting for the threads to join...");
    while let Some(h) = handles.pop() {
        h.join().unwrap();
    }
    info!("Threads joined!");

    pbar::finish_progress_bar();

    //info!("Degrees: {:?}", degree_counters);

    info!("Writing degree csv...");
    {
        let mut wtr = csv::Writer::from_path(&app.output_degrees).unwrap();
        wtr.write_record(&["node_id", "degree"]).unwrap();
        for (i, j) in degree_counters.iter().enumerate() {
            wtr.write_record(&[format!("{}", i), format!("{}", *j)]).unwrap();
        }
        wtr.flush().unwrap();
    }
    info!("Done writing!");

    info!("Writing degree txt...");
    {
        let mut f = File::create(app.output_degrees_txt).expect("Unable to create file");
        for i in degree_counters.iter() {
            write!(f, "{}\n", i).unwrap();
        }
        f.flush().unwrap();
    }
    info!("Done writing!");


    info!("Writing degree distribution csv...");
    {
        let mut wtr = csv::Writer::from_path(&app.output_degrees_distribution).unwrap();
        wtr.write_record(&["x", "number of nodes with degree > x / number of nodes"]).unwrap();
        for x in 0..=degree_counters.len() {
            let s: f64 = degree_counters.iter().filter(|&d| *d > x as usize).count() as f64;
            let v = s / (degree_counters.len() as f64);
            let x = x as f64;
            wtr.write_record(&[format!("{}", x), format!("{}", v)]).unwrap();
        }
        wtr.flush().unwrap();
    }
    info!("Done writing!");

    //plot::main(degree_counters);
}