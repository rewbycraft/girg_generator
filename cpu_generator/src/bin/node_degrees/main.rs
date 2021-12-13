use std::path::PathBuf;
use clap::{Parser, ValueHint};
use tracing::info;

pub mod plot;

/// Simple program to greet a person
#[derive(Parser, Debug)]
struct Args {
    /// Number of vertices
    #[clap(short, long, default_value_t = 4)]
    workers: usize,
    /// Number of vertices
    #[clap(long, default_value_t = 4000)]
    block_size: u64,
    /// Number of vertices
    #[clap(short, long, default_value_t = 100000)]
    vertices: u64,
    /// Alpha value of the probability function
    #[clap(short, long, default_value_t = 1.0)]
    alpha: f32,
    /// Alpha value of the pareto distribution
    #[clap(short, long, default_value_t = 1.5)]
    beta: f32,
    /// x_min value of the pareto distribution
    #[clap(short, long, default_value_t = 1.0)]
    x_min: f32,
    #[clap(short, long, parse(from_os_str), value_hint = ValueHint::FilePath, default_value = "/tmp/degrees_distribution.csv")]
    /// File to write items to
    pub output: PathBuf,
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
    std::env::set_var(
        "RUST_LOG",
        std::env::var("RUST_LOG").unwrap_or("info".to_string()),
    );
    tracing_subscriber::fmt::fmt().init();

    let app: Args = Args::parse();
    let params = app.get_params();

    info!("Params:\n{:#?}", params);

    let (block_sender, block_receiver) = crossbeam_channel::bounded(5);
    let (edge_sender, edge_receiver) = crossbeam_channel::bounded(5);

    let mut handles = cpu_generator::threads::start_workers(app.workers, edge_sender, block_receiver, &params);
    handles.push(cpu_generator::threads::start_generate_blocks_thread(block_sender, app.block_size, &params));

    let mut degree_counters: Vec<usize> = Vec::new();
    degree_counters.resize(app.vertices as usize, 0usize);

    info!("Receiving edges...");
    for edge_block in edge_receiver {
        for (i, _) in edge_block {
            *degree_counters.get_mut(i as usize).unwrap() += 1;
        }
    }
    info!("All edges received!");
    info!("Degrees: {:?}", degree_counters);

    info!("Waiting for the threads to join...");
    while let Some(h) = handles.pop() {
        h.join().unwrap();
    }
    info!("Threads joined!");

    info!("Writing degree distribution csv...");
    let mut wtr = csv::Writer::from_path(&app.output).unwrap();
    for x in 0..=degree_counters.len() {
        let s: f64 = degree_counters.iter().filter(|&d| *d > x as usize).count() as f64;
        let v = s / (degree_counters.len() as f64);
        let x = x as f64;
        wtr.write_record(&[format!("{}", x), format!("{}", v)]).unwrap();
        println!("{},{}", x, v);
    }
    wtr.flush().unwrap();
    info!("Done writing!");

    //plot::main(degree_counters);
}