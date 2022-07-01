//! This module holds the CPU extensions to the parameters of the graph being generated.
//! See [`generator_core::params`] for more information.

use std::ops::{Deref, DerefMut};
use fixed_size_buffer::cpu::FixedSizeCPUBuffer;
use fixed_size_buffer::GetFixedSizeBufferRef;

pub use generator_core::params::*;

use crate::algorithm::generate_parameters;
use crate::random;

/// Instance of [`GenerationParameters`] with the seeds stored on the CPU.
#[derive(Debug)]
pub struct CPUGenerationParameters<'a> {
    /// Parameters for the graph.
    pub params: GenerationParameters<'a>,
    /// Backing storage for the seeds.
    pub real_seeds: FixedSizeCPUBuffer<'a, u64>,
}

impl<'a> CPUGenerationParameters<'a> {
    /// Clone implementation that creates a new lifetime.
    pub fn clone<'b>(&self) -> CPUGenerationParameters<'b> {
        let real_seeds = self.real_seeds.clone();
        let params = self.params.clone().replace_seeds(real_seeds.get_ref());
        CPUGenerationParameters {
            params,
            real_seeds,
        }
    }

    /// Generate the list of weights for all nodes.
    pub fn compute_weights(&self) -> Vec<f32> {
        (0..self.v).map(|j| self.compute_weight(j)).collect()
    }

    /// Compute the position of a node.
    ///
    /// # Arguments
    /// * `j` - The node index.
    pub fn compute_position(&self, j: u64) -> Vec<f32> {
        (0..self.num_dimensions())
            .map(|d| self.compute_property(j, SeedEnum::Dimension(d)))
            .collect()
    }

    /// Compute the positions of all nodes.
    pub fn compute_positions(&self) -> Vec<Vec<f32>> {
        (0..self.v).map(|j| self.compute_position(j)).collect()
    }

    /// Compute the weights and positions of all nodes in an interleaved fashion.
    ///
    /// The output will be in the format `[node 0 weight, node 0 position 0, node 0 position 1, ..., node 1 weight, node 1 position 0, node 1 position 1, ..., ...]`.
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

    /// Create an iterator over all the tiles in this graph.
    pub fn tiles(&self) -> Box<dyn Iterator<Item=crate::tiles::Tile>> {
        Box::new(
            crate::tiles::TilesIterator::new(self.v, self.tile_size)
                .skip(self.shard_index)
                .step_by(self.shard_count),
        )
    }

    /// Number of tiles in this graph.
    pub fn num_tiles(&self) -> u64 {
        num_integer::div_ceil(self.v, self.tile_size).pow(2) / (self.shard_count as u64)
    }

    /// Create new instance of parameters with random seeds.
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

    /// Create new instance of parameters with known seeds.
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

        let params: GenerationParameters = GenerationParameters {
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