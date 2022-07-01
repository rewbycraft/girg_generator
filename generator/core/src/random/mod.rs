//! Hash-based psuedo-random number generation

pub mod murmur3;

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use no_std_compat::prelude::v1::*;

/// Parameters for a Pareto distribution.
#[derive(Clone, Debug, Copy)]
#[cfg_attr(
    all(not(target_os = "cuda"), feature = "gpu"),
    derive(cust::DeviceCopy)
)]
pub struct ParetoDistribution {
    /// x_min value of the distribution
    x: f32,
    /// alpha value of the distribution
    alpha: f32,
}

impl ParetoDistribution {
    /// Create new distribution.
    ///
    /// # Arguments
    /// * `x` - The `x_min` value for the pareto distribution.
    /// * `alpha` - The `alpha` value for the pareto distribution.
    pub fn new(x: f32, alpha: f32) -> Self {
        ParetoDistribution { x, alpha }
    }

    /// Convert a uniform random value in the range `[0,1]` to a pareto random value.
    pub fn convert_uniform(&self, u: f32) -> f32 {
        self.x / ((1.0f32 - u).powf(1.0f32 / self.alpha))
    }
}

/// Compute a random property for a node given a seed.
///
/// Returns a value in the range `[0.0f32, 1.0f32]`.
///
/// You should usually not call this directly.
/// Instead, call this via the [`GenerationParameters::compute_property()`](crate::params::GenerationParameters::compute_property()) method.
///
/// # Arguments
/// * `i` - The node for which to compute the property.
/// * `seed` - The seed number to use.
///
pub fn random_property(i: u64, seed: u64) -> f32 {
    let h = murmur3::murmur3_32_2(i, seed);
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    v as f32
}

/// Compute the edge random number.
///
/// Returns a value in the range `[0.0f32, 1.0f32]`.
///
/// You should usually not call this directly.
/// Instead, call this via the [`GenerationParameters::edge_random()`](crate::params::GenerationParameters::edge_random()) method.
///
/// # Arguments
/// * `i` - The left node of the edge for which to compute the number.
/// * `j` - The right node of the edge for which to compute the number.
/// * `seed` - The seed number to use.
pub fn random_edge(i: u64, j: u64, seed: u64) -> f32 {
    let h = murmur3::murmur3_32_3(i, j, seed);
    let h = h as f64;
    let v = h / (u32::MAX as f64);
    v as f32
}
