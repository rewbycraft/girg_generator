#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(clippy::missing_safety_doc)]

#[cfg(target_os = "cuda")]
use cuda_std::GpuFloat;
use no_std_compat::cmp::Ordering::Equal;

use crate::params::{GenerationParameters, SeedEnum, SeedGettable};

#[cfg(not(target_os = "cuda"))]
pub mod generator;
pub mod params;
pub mod random;
#[cfg(not(target_os = "cuda"))]
pub mod threads;
#[cfg(not(target_os = "cuda"))]
pub mod tiles;

trait PositionGetter {
    fn get_position(&self, node: u64, dimension: usize) -> f32;
}

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

pub fn compute_probability<S: SeedGettable + Sized>(
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

//32 bit float epsilon on NVidia GPUs
//See <cuda root>/targets/x86_64-linux/include/CL/cl_platform.h line 192
//const EPS: f32 = 1.1920928955078125e-7;
//const EPS: f32 = 9.765625e-04;
//const EPS: f32 = 2.384185791015625e-07;

pub fn generate_edge<S: SeedGettable + Sized>(
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

    //Compute goal: p > rp

    //Since nvidia GPUs have an epsilon much larger than the one of the x86_64 cpu,
    // we need to compare the difference to that epsilon.
    //On the GPU this is effectively saying p is at least one representable number away from rp.
    //In other words, p > rp.
    //On the CPU we simulate the effect of the larger epsilon in order to get the exact same numbers.
    // let diff = p - rp;
    // diff > EPS
}
