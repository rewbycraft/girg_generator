//! Hash-based psuedo-random number generation

pub mod murmur3;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use no_std_compat::prelude::v1::*;

#[derive(Clone, Debug, Copy)]
#[cfg_attr(
    all(not(target_os = "cuda"), feature = "gpu"),
    derive(cust::DeviceCopy)
)]
pub struct ParetoDistribution {
    /// x_min
    pub x: f32,
    /// alpha value of the distribution
    pub alpha: f32,
}

impl ParetoDistribution {
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
    v as f32
}
