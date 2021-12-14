use std::io::prelude::*;
use std::fs::File;

fn main() {
    let pareto = cpu_generator::random::ParetoDistribution::new(1.0f32, 1.5f32);
    let params = cpu_generator::GenerationParameters::new(pareto, 1.5f32, 100000);
    let mut f = File::create("weights.txt").expect("Unable to create file");
    for i in params.compute_weights() {
        write!(f, "{}", i).unwrap();
    }
    f.flush().unwrap();
}