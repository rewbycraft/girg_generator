//! The probability function module.

//This import isn't actually unused, the compiler just gets confused.
//It's needed for float intrinsics.
#[allow(unused_imports)]
use cuda_std::GpuFloat;

use crate::params::{GenerationParameters, SeedEnum, SeedGettable};
use crate::random;
use no_std_compat::cmp::Ordering::Equal;

/// Computes the distance between two positions of an d-dimensional torus.
///
/// This function assumes the length of the slices is equal to the number of dimensions and that both slices are of equal length.
pub fn compute_distance(p_i: &[f32], p_j: &[f32]) -> f32 {
    fn dist_c(i: f32, j: f32) -> f32 {
        (i - j).abs().min(1.0f32 - ((i - j).abs()))
    }
    p_i.iter()
        .zip(p_j.iter())
        .map(|(p0, p1)| dist_c(*p0, *p1))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
        .unwrap()
}

/// Actual probability function.
///
/// # Arguments
/// * `d` - Distance between nodes i and j.
/// * `w_i` - Weight of node i.
/// * `w_j` - Weight of node j.
/// * `params` - Reference to the parameters for the graph being generated. See [GenerationParameters].
pub fn compute_probability<S: SeedGettable>(
    // Distance between nodes i and j.
    d: f32,
    w_i: f32,
    w_j: f32,
    params: &GenerationParameters<S>,
) -> f32 {
    if params.alpha.is_infinite() {
        let v = ((w_i * w_j) / params.w).powf(1.0f32 / params.num_dimensions() as f32);
        if d <= v {
            1.0f32
        } else {
            0.0f32
        }
    } else {
        (
            //The main multiplication
            (((w_i * w_j) / params.w).powf(params.alpha))
                // 1/dist^(ad)
                / (d.powf(params.alpha * params.num_dimensions() as f32))
        )
        .min(1.0f32)
    }
}

/// Function that determines whether an edge exists.
///
/// # Arguments
/// * `i` - Left node index.
/// * `j` - Right node index.
/// * `w_i` - Weight of node i.
/// * `w_j` - Weight of node j.
/// * `p_i` - Position of node i.
/// * `p_j` - Position of node j.
/// * `params` - Reference to the parameters for the graph being generated. See [GenerationParameters].
pub fn generate_edge<S: SeedGettable>(
    i: u64,
    j: u64,
    w_i: f32,
    w_j: f32,
    p_i: &[f32],
    p_j: &[f32],
    params: &GenerationParameters<S>,
) -> bool {
    let d = compute_distance(p_i, p_j);
    let p = compute_probability(d, w_i, w_j, params);
    let rp = random::random_edge(i, j, params.get_seed(SeedEnum::Edge));

    p > rp
}
