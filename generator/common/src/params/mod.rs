use std::ops::{Deref, DerefMut};
use fixed_size_buffer::cpu::FixedSizeCPUBuffer;
use fixed_size_buffer::GetFixedSizeBufferRef;

pub use generator_core::params::*;

use crate::algorithm::generate_parameters;
use crate::random;

#[derive(Debug)]
pub struct CPUGenerationParameters<'a> {
    pub params: GenerationParameters<'a>,
    pub real_seeds: FixedSizeCPUBuffer<'a, u64>,
}

// impl<'a> Clone for CPUGenerationParameters<'a> {
//     fn clone(&self) -> CPUGenerationParameters<'_> {
//         let mut real_seeds = self.real_seeds.clone();
//         let mut params = self.params.clone();
//         params.seeds = real_seeds.get_ref();
//         CPUGenerationParameters {
//             params,
//             real_seeds,
//         }
//     }
// }

impl<'a> CPUGenerationParameters<'a> {
    pub fn clone<'b>(&self) -> CPUGenerationParameters<'b> {
        let mut real_seeds = self.real_seeds.clone::<'b>();
        let mut params = self.params.clone().replace_seeds(real_seeds.get_ref());
        CPUGenerationParameters {
            params,
            real_seeds,
        }
    }

    pub fn compute_weights(&self) -> Vec<f32> {
        (0..self.v).map(|j| self.compute_weight(j)).collect()
    }

    pub fn compute_position(&self, j: u64) -> Vec<f32> {
        (0..self.num_dimensions())
            .map(|d| self.compute_property(j, SeedEnum::Dimension(d)))
            .collect()
    }

    pub fn compute_positions(&self) -> Vec<Vec<f32>> {
        (0..self.v).map(|j| self.compute_position(j)).collect()
    }

    pub fn compute_interleaved_variables(&self) -> Vec<f32> {
        (0..self.v)
            .flat_map(|j| {
                vec![self.compute_weight(j)]
                    .into_iter()
                    .chain(
                        (0..self.num_dimensions())
                            .map(|d| self.compute_property(j, SeedEnum::Dimension(d))),
                    )
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    pub fn tiles(&self) -> Box<dyn Iterator<Item=crate::tiles::Tile>> {
        Box::new(
            crate::tiles::TilesIterator::new(self.v, self.tile_size)
                .skip(self.shard_index)
                .step_by(self.shard_count),
        )
    }

    pub fn num_tiles(&self) -> u64 {
        num_integer::div_ceil(self.v, self.tile_size).pow(2) / (self.shard_count as u64)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_dimensions: usize,
        pareto: random::ParetoDistribution,
        alpha: f32,
        v: u64,
        tile_size: u64,
        edgebuffer_size: u64,
        pregenerate_numbers: bool,
        gpu_blocks: u32,
        shard_index: usize,
        shard_count: usize,
    ) -> Self {
        let seeds: Vec<u64> = random::generate_seeds(num_dimensions + 2);

        Self::from_seeds(
            num_dimensions,
            pareto,
            alpha,
            v,
            &seeds,
            tile_size,
            edgebuffer_size,
            pregenerate_numbers,
            gpu_blocks,
            shard_index,
            shard_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_seeds(
        num_dimensions: usize,
        pareto: random::ParetoDistribution,
        alpha: f32,
        v: u64,
        seeds: &[u64],
        tile_size: u64,
        edgebuffer_size: u64,
        pregenerate_numbers: bool,
        gpu_blocks: u32,
        shard_index: usize,
        shard_count: usize,
    ) -> Self {
        if seeds.len() != num_dimensions + 2 {
            panic!(
                "Invalid seeds length: {} != {}",
                seeds.len(),
                num_dimensions + 2
            );
        }

        let real_seeds = Vec::from(seeds);
        let real_seeds = FixedSizeCPUBuffer::from(real_seeds);

        let mut params: GenerationParameters = GenerationParameters {
            seeds: real_seeds.get_ref(),
            pregenerate_numbers,
            gpu_blocks,
            dims: num_dimensions,
            pareto,
            alpha,
            w: 0.0,
            v,
            tile_size,
            edgebuffer_size,
            shard_index,
            shard_count,
        };

        let mut s = Self {
            params,
            real_seeds,
        };

        // Initialize
        generate_parameters(&mut s);

        s
    }
}

impl<'a> Deref for CPUGenerationParameters<'a> {
    type Target = GenerationParameters<'a>;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl<'a> DerefMut for CPUGenerationParameters<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.params
    }
}