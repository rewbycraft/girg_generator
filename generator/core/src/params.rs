//! This module holds the parameters of the graph being generated.
//!
//! See [GenerationParameters] for the explanation of how this is structured.

use no_std_compat::ops::Div;
use no_std_compat::prelude::v1::*;

use super::random;

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub enum SeedEnum {
    Weight,
    Edge,
    Dimension(usize),
}

pub trait SeedGettable: Sized {
    fn get_seed(&self, s: SeedEnum) -> u64;
}

#[derive(Clone, Copy, Debug)]
pub struct RawSeeds {
    seeds: *const u64,
}

impl RawSeeds {
    pub fn new(seeds: *const u64) -> Self {
        RawSeeds { seeds }
    }
}

#[cfg(all(not(target_os = "cuda"), feature = "gpu"))]
unsafe impl cust::memory::DeviceCopy for RawSeeds {}

impl SeedGettable for RawSeeds {
    fn get_seed(&self, s: SeedEnum) -> u64 {
        let i = match s {
            SeedEnum::Weight => 0,
            SeedEnum::Edge => 1,
            SeedEnum::Dimension(i) => 2 + i,
        };
        //TODO: Add safety checks.
        unsafe { *self.seeds.add(i as usize) }
    }
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    all(not(target_os = "cuda"), feature = "gpu"),
    derive(cust::DeviceCopy)
)]
/// This struct holds the main parameters for the graph being generated.
///
/// An important requirement for this struct is that it is [Copy]-able.
/// This is needed because the [GPU generator](generator_gpu) needs to be able to copy this into the GPU VRAM.
///
/// # Why are the seeds a generic type argument?
///
/// The GPU generator imposes an interesting challenge.
/// Since the number of dimensions is not known until run-time, the number of seeds is not either.
/// As such, a [Vec<u64>] is used to hold the seeds on the CPU side of the program.
///
/// But we cannot just copy such a [Vec<u64>] to the GPU.
/// Instead the data must be uploaded separately and a raw pointer (`*const u64`) must be stored for
pub struct GenerationParameters<S: SeedGettable + Sized> {
    pub seeds: S,
    pub pregenerate_numbers: bool,
    pub gpu_blocks: u32,
    pub dims: usize,
    pub pareto: random::ParetoDistribution,
    pub alpha: f32,
    pub w: f32,
    pub v: u64,
    pub tile_size: u64,
    pub edgebuffer_size: u64,
    pub shard_index: usize,
    pub shard_count: usize,
}

impl<S: SeedGettable + Sized> GenerationParameters<S> {
    pub fn num_dimensions(&self) -> usize {
        self.dims
    }

    pub fn get_seed(&self, s: SeedEnum) -> u64 {
        self.seeds.get_seed(s)
    }

    pub fn compute_weight(&self, j: u64) -> f32 {
        random::uniform_to_pareto(self.compute_property(j, SeedEnum::Weight), &self.pareto)
    }

    pub fn compute_property(&self, j: u64, p: SeedEnum) -> f32 {
        random::random_property(j, self.get_seed(p))
    }

    pub fn fill_dims(&self, j: u64, p: &mut [f32]) {
        for (d, p) in p.iter_mut().enumerate() {
            *p = self.compute_property(j, SeedEnum::Dimension(d));
        }
    }

    pub fn pos_to_tile(&self, x: u64, y: u64) -> ((u64, u64), (u64, u64)) {
        let bx = x.div(self.tile_size) * self.tile_size;
        let by = y.div(self.tile_size) * self.tile_size;
        let ex = bx + self.tile_size;
        let ey = by + self.tile_size;
        ((bx, by), (ex, ey))
    }
}
