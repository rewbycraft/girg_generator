//! This module holds the parameters of the graph being generated.
//!
//! See [GenerationParameters] for the explanation of how this is structured.
//!
//! Note that not all functions are available in here.
//! Some of them live in the common module where the CPU based extensions are implemented.

use no_std_compat::ops::Div;
use no_std_compat::prelude::v1::*;
use fixed_size_buffer::FixedSizeBufferRef;
use crate::tiles::Tile;

use super::random;

/// Enumeration of properties that derive from a seed.
#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq)]
pub enum SeedEnum {
    /// The weight of a node.
    Weight,
    /// The random value that determines whether an edge exists.
    Edge,
    /// The position of a node in one of the dimensions.
    Dimension(usize),
}

impl SeedEnum {
    fn seed_index(&self) -> usize {
        match self {
            SeedEnum::Weight => 0,
            SeedEnum::Edge => 1,
            SeedEnum::Dimension(i) => 2 + *i,
        }
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
/// This is needed because the GPU generator needs to be able to copy this into the GPU VRAM.
pub struct GenerationParameters<'a> {
    /// Reference to a fixed size buffer that holds the random seeds for this run.
    pub seeds: FixedSizeBufferRef<'a, u64>,
    /// Whether to pre-generate the random numbers or use on-demand number generation.
    pub pregenerate_numbers: bool,
    /// The amount of thread blocks to use on the GPU. Effectively controls GPU concurrency.
    pub gpu_blocks: u32,
    /// Number of dimensions in the space where vertices exist.
    pub dims: usize,
    /// The random number distribution to use.
    pub pareto: random::ParetoDistribution,
    /// The alpha value to use. (For the GIRG probability function.)
    pub alpha: f32,
    /// Sum of all weights
    pub w: f32,
    /// Number of vertices
    pub v: u64,
    /// Size of the processing tiles
    pub tile_size: u64,
    /// Size of the edge-buffer that stores output
    pub edgebuffer_size: u64,
    /// Which of the shards this process is
    pub shard_index: usize,
    /// How many shards exist
    pub shard_count: usize,
}

impl<'a> GenerationParameters<'a> {
    /// Replaces the seeds buffer with another one.
    pub fn replace_seeds<'b>(self, seeds: FixedSizeBufferRef<'b, u64>) -> GenerationParameters<'b> {
        GenerationParameters {
            seeds,
            pregenerate_numbers: self.pregenerate_numbers,
            gpu_blocks: self.gpu_blocks,
            dims: self.dims,
            pareto: self.pareto,
            alpha: self.alpha,
            w: self.w,
            v: self.v,
            tile_size: self.tile_size,
            edgebuffer_size: self.edgebuffer_size,
            shard_index: self.shard_index,
            shard_count: self.shard_count,
        }
    }

    /// Number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.dims
    }

    /// Get a seed from the buffer
    pub fn get_seed(&self, s: SeedEnum) -> u64 {
        self.seeds[s.seed_index()]
    }

    /// Compute the weight of a node.
    ///
    /// # Arguments
    /// * `j` - The index of the node for which we want to compute the weight. (`0 <= j < self.v`)
    pub fn compute_weight(&self, j: u64) -> f32 {
        self.compute_property(j, SeedEnum::Weight)
    }

    /// Computes the value of a node property.
    ///
    /// # Arguments
    /// * `j` - The index of the node for which we want to compute the property. (`0 <= j < self.v`)
    /// * `p` - The node property we want to compute.
    pub fn compute_property(&self, j: u64, p: SeedEnum) -> f32 {
        let v = random::random_property(j, self.get_seed(p));
        match p {
            SeedEnum::Weight => self.pareto.convert_uniform(v),
            // Edges are edge properties, not node properties,
            SeedEnum::Edge => panic!("cannot compute node property of type EDGE"),
            _ => v,
        }
    }

    /// Compute the random value for an edge.
    ///
    /// # Arguments
    /// * `i` - The index of the left node of the edge.
    /// * `j` - The index of the right node of the edge.
    pub fn edge_random(&self, i: u64, j: u64) -> f32 {
        random::random_edge(i, j, self.get_seed(SeedEnum::Edge))
    }

    /// Fill the position array of a node.
    ///
    /// Compute the position of the node and write it to existing array `p`.
    /// Assumes `p.len() == self.num_dimensions()`
    ///
    /// # Arguments
    /// * `j` - The index of the node.
    /// * `p` - The array to write the position into.
    pub fn fill_dims(&self, j: u64, p: &mut [f32]) {
        for (d, p) in p.iter_mut().enumerate() {
            *p = self.compute_property(j, SeedEnum::Dimension(d));
        }
    }

    /// Compute which tile contains the given position on in the adjacency matrix.
    ///
    /// # Arguments
    /// * `x` - The left node of the edge at this position.
    /// * `y` - The right node of the edge at this position.
    pub fn pos_to_tile(&self, x: u64, y: u64) -> Tile {
        let bx = x.div(self.tile_size) * self.tile_size;
        let by = y.div(self.tile_size) * self.tile_size;
        let ex = bx + self.tile_size - 1;
        let ey = by + self.tile_size - 1;
        Tile((bx, by), (ex.min(self.v - 1), ey.min(self.v - 1)))
    }
}
