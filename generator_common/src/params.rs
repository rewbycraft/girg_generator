#[cfg(not(target_os = "cuda"))]
use std::ops::Div;

#[cfg(target_os = "cuda")]
use no_std_compat::ops::Div;
#[cfg(target_os = "cuda")]
use no_std_compat::prelude::v1::*;

use super::random;

pub const DIMENSIONS: usize = 2;

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq, Hash)]
pub enum SeedEnum {
    Weight,
    Edge,
    Dimension(usize),
}

#[derive(Clone, Debug, Copy)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
pub struct GenerationParameters {
    seeds: [u64; DIMENSIONS + 2],
    pub pareto: random::ParetoDistribution,
    pub alpha: f32,
    pub w: f32,
    pub v: u64,
    pub tile_size: u64,
}

// #[cfg(not(target_os = "cuda"))]
// unsafe impl cust::memory::DeviceCopy for GenerationParameters {}

impl GenerationParameters {
    pub fn num_dimensions(&self) -> usize {
        DIMENSIONS
    }

    pub fn get_seed(&self, s: SeedEnum) -> u64 {
        let i = match s {
            SeedEnum::Weight => 0,
            SeedEnum::Edge => 1,
            SeedEnum::Dimension(i) => 2 + i,
        };
        self.seeds[i]
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn new(pareto: random::ParetoDistribution, alpha: f32, v: u64, tile_size: u64) -> Self {
        let seeds: [u64; DIMENSIONS+2] = random::generate_seeds();

        Self::from_seeds(pareto, alpha, v, &seeds, tile_size)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn from_seeds(pareto: random::ParetoDistribution, alpha: f32, v: u64, seeds: &[u64], tile_size: u64) -> Self {
        if seeds.len() != DIMENSIONS + 2 {
            panic!("Invalid seeds length: {} != {}", seeds.len(), DIMENSIONS + 2);
        }

        let mut ss = [0u64; DIMENSIONS + 2];

        ss.copy_from_slice(seeds);

        let mut s = Self {
            seeds: ss,
            pareto,
            alpha,
            w: 0.0,
            v,
            tile_size,
        };

        s.compute_w();
        s
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_w(&mut self) {
        use tracing::info;

        info!("Computing W...");
        self.w = self.compute_weights().into_iter().sum();
        info!("Computed W = {}", self.w);
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_weight(&self, j: u64) -> f32 {
        random::uniform_to_pareto(random::random_property(j, self.get_seed(SeedEnum::Weight)), &self.pareto)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_weights(&self) -> Vec<f32> {
        (0..self.v).map(|j| self.compute_weight(j)).collect()
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_position(&self, j: u64) -> [f32; DIMENSIONS] {
        let mut a = [0f32; DIMENSIONS];
        for d in 0..DIMENSIONS {
            a[d] = random::random_property(j, self.get_seed(SeedEnum::Dimension(d)));
        }
        a
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_positions(&self) -> Vec<[f32; DIMENSIONS]> {
        (0..self.v).map(|j| self.compute_position(j)).collect()
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_interleaved_variables(&self) -> Vec<f32> {
        (0..self.v)
            .flat_map(|j| {
                vec![self.compute_weight(j)]
                    .into_iter()
                    .chain((0..DIMENSIONS)
                        .map(|d|
                            random::random_property(j, self.get_seed(SeedEnum::Dimension(d)))
                        ))
                    .collect::<Vec<f32>>()
            }).collect()
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn tiles(&self) -> super::tiles::TilesIterator {
        super::tiles::TilesIterator::new(self.v, self.tile_size)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn num_tiles(&self) -> u64 {
        num_integer::div_ceil(self.v, self.tile_size).pow(2)
    }

    pub fn pos_to_tile(&self, x: u64, y: u64) -> ((u64, u64), (u64, u64)) {
        let bx = x.div(self.tile_size) * self.tile_size;
        let by = y.div(self.tile_size) * self.tile_size;
        let ex = bx + self.tile_size;
        let ey = by + self.tile_size;
        ((bx, by), (ex, ey))
    }
}