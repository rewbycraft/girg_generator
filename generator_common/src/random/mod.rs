pub mod murmur3;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
#[cfg(target_os = "cuda")]
use no_std_compat::prelude::v1::*;

#[derive(Clone, Debug, Copy)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
pub struct ParetoDistribution {
    pub x: f32,
    pub alpha: f32,
}

// #[cfg(not(target_os = "cuda"))]
// unsafe impl cust::memory::DeviceCopy for ParetoDistribution {}

impl ParetoDistribution {
    #[cfg(not(target_os = "cuda"))]
    pub fn new(x: f32, alpha: f32) -> Self {
        ParetoDistribution { x, alpha }
    }
}

pub fn uniform_to_pareto(u: f32, dist: &ParetoDistribution) -> f32 {
    dist.x / ((1.0f32 - u).powf(1.0f32 / dist.alpha))
}

pub fn random_property(i: u64, seed: u64) -> f32 {
    let h = murmur3::murmur3_32_2(i, seed);
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    v as f32
}

pub fn random_edge(i: u64, j: u64, seed: u64) -> f32 {
    let h = murmur3::murmur3_32_3(i, j, seed);
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    let p = v as f32;
    p
}

#[cfg(not(target_os = "cuda"))]
pub fn generate_seeds(n: usize) -> Vec<u64> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut seeds = vec![];
    seeds.resize(n, 0);

    for i in 0..n {
        loop {
            let r: u64 = rng.gen();
            if (i == 0) || (!(seeds[0..(i-1)].contains(&r))) {
                seeds[i] = r;
                break;
            }
        }
    }

    seeds
}
