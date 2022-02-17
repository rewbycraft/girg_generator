use generator_core::params::{GenerationParameters, SeedEnum};
use crate::algorithm::generate_parameters;
use crate::params::VecSeeds;
use crate::random;

pub trait GenerationParametersExt {
    fn compute_weights(&self) -> Vec<f32>;
    fn compute_position(&self, j: u64) -> Vec<f32>;
    fn compute_positions(&self) -> Vec<Vec<f32>>;
    fn compute_interleaved_variables(&self) -> Vec<f32>;
    fn tiles(&self) -> Box<dyn Iterator<Item = crate::tiles::Tile>>;
    fn num_tiles(&self) -> u64;
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> Self;

    #[allow(clippy::too_many_arguments)]
    fn from_seeds(
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
    ) -> Self;
}

impl GenerationParametersExt for GenerationParameters<VecSeeds> {
    fn compute_weights(&self) -> Vec<f32> {
        (0..self.v).map(|j| self.compute_weight(j)).collect()
    }

    fn compute_position(&self, j: u64) -> Vec<f32> {
        (0..self.num_dimensions())
            .map(|d| self.compute_property(j, SeedEnum::Dimension(d)))
            .collect()
    }

    fn compute_positions(&self) -> Vec<Vec<f32>> {
        (0..self.v).map(|j| self.compute_position(j)).collect()
    }

    fn compute_interleaved_variables(&self) -> Vec<f32> {
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

    fn tiles(&self) -> Box<dyn Iterator<Item = crate::tiles::Tile>> {
        Box::new(crate::tiles::TilesIterator::new(self.v, self.tile_size)
            .skip(self.shard_index)
            .step_by(self.shard_count))
    }

    fn num_tiles(&self) -> u64 {
        num_integer::div_ceil(self.v, self.tile_size).pow(2) / (self.shard_count as u64)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
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
        let seeds: Vec<u64> = crate::random::generate_seeds(num_dimensions + 2);

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
    fn from_seeds(
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

        let mut s = Self {
            seeds: VecSeeds {
                seeds: Vec::from(seeds),
            },
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

        // Initialize
        generate_parameters(&mut s);

        s
    }
}
