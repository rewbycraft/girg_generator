#[cfg(not(target_os = "cuda"))]
use std::ops::Div;

#[cfg(target_os = "cuda")]
use no_std_compat::ops::Div;
#[cfg(target_os = "cuda")]
use no_std_compat::prelude::v1::*;

use super::random;

#[derive(Clone, Copy, Eq, Ord, PartialOrd, PartialEq, Hash)]
pub enum SeedEnum {
    Weight,
    Edge,
    Dimension(usize),
}

pub trait SeedGettable {
    fn get_seed(&self, s: SeedEnum) -> u64;
}

#[cfg(not(target_os = "cuda"))]
#[derive(Debug, Clone)]
pub struct VecSeeds {
    seeds: Vec<u64>,
}

#[cfg(not(target_os = "cuda"))]
impl SeedGettable for VecSeeds {
    fn get_seed(&self, s: SeedEnum) -> u64 {
        let i = match s {
            SeedEnum::Weight => 0,
            SeedEnum::Edge => 1,
            SeedEnum::Dimension(i) => 2 + i,
        };
        self.seeds[i]
    }
}

#[cfg(not(target_os = "cuda"))]
impl VecSeeds {
    #[cfg(not(target_os = "cuda"))]
    pub fn get_dbuffer(&self) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>> {
        cust::memory::DeviceBuffer::from_slice(&self.seeds)
    }
    #[cfg(not(target_os = "cuda"))]
    pub unsafe fn get_dbuffer_async(
        &self,
        stream: &cust::stream::Stream,
    ) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>> {
        cust::memory::DeviceBuffer::from_slice_async(&self.seeds, stream)
    }
}

//#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Debug, Clone, Copy)]
pub struct RawSeeds {
    seeds: *const u64,
}

#[cfg(not(target_os = "cuda"))]
unsafe impl cust::memory::DeviceCopy for RawSeeds {}

impl SeedGettable for RawSeeds {
    fn get_seed(&self, s: SeedEnum) -> u64 {
        let i = match s {
            SeedEnum::Weight => 0,
            SeedEnum::Edge => 1,
            SeedEnum::Dimension(i) => 2 + i,
        };
        unsafe { *self.seeds.add(i as usize) }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy, Copy))]
pub struct GenerationParameters<S: SeedGettable + Sized> {
    pub seeds: S,
    pub pregenerate_numbers: bool,
    pub gpu_blocks: u32,
    dims: usize,
    pub pareto: random::ParetoDistribution,
    pub alpha: f32,
    pub w: f32,
    pub v: u64,
    pub tile_size: u64,
    pub edgebuffer_size: u64,
    pub shard_index: usize,
    pub shard_count: usize,
}

impl GenerationParameters<RawSeeds> {
    #[cfg(not(target_os = "cuda"))]
    pub fn from_vecseeds_and_device_ptr(
        vseeds: &GenerationParameters<VecSeeds>,
        device_ptr: cust::memory::DevicePointer<u64>,
    ) -> Self {
        Self {
            seeds: RawSeeds {
                seeds: device_ptr.as_raw(),
            },
            pregenerate_numbers: vseeds.pregenerate_numbers,
            gpu_blocks: vseeds.gpu_blocks,
            dims: vseeds.dims,
            pareto: vseeds.pareto,
            alpha: vseeds.alpha,
            w: vseeds.w,
            v: vseeds.v,
            tile_size: vseeds.tile_size,
            edgebuffer_size: vseeds.edgebuffer_size,
            shard_index: vseeds.shard_index,
            shard_count: vseeds.shard_count,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
impl GenerationParameters<VecSeeds> {
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

        s.compute_w();
        s
    }
}

// #[cfg(not(target_os = "cuda"))]
// unsafe impl cust::memory::DeviceCopy for GenerationParameters {}

impl<S: SeedGettable + Sized> GenerationParameters<S> {
    pub fn num_dimensions(&self) -> usize {
        self.dims
    }

    pub fn get_seed(&self, s: SeedEnum) -> u64 {
        self.seeds.get_seed(s)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_w(&mut self) {
        use tracing::info;

        info!("Computing W...");
        self.w = self.compute_weights().into_iter().sum();
        info!("Computed W = {}", self.w);
    }

    pub fn compute_weight(&self, j: u64) -> f32 {
        random::uniform_to_pareto(self.compute_property(j, SeedEnum::Weight), &self.pareto)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_weights(&self) -> Vec<f32> {
        (0..self.v).map(|j| self.compute_weight(j)).collect()
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_position(&self, j: u64) -> Vec<f32> {
        (0..self.num_dimensions())
            .map(|d| self.compute_property(j, SeedEnum::Dimension(d)))
            .collect()
    }

    fn compute_property(&self, j: u64, p: SeedEnum) -> f32 {
        random::random_property(j, self.get_seed(p))
    }

    pub fn fill_dims(&self, j: u64, p: &mut [f32]) {
        for (d, p) in p.iter_mut().enumerate() {
            *p = self.compute_property(j, SeedEnum::Dimension(d));
        }
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn compute_positions(&self) -> Vec<Vec<f32>> {
        (0..self.v).map(|j| self.compute_position(j)).collect()
    }

    #[cfg(not(target_os = "cuda"))]
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

    #[cfg(not(target_os = "cuda"))]
    pub fn tiles(&self) -> impl Iterator<Item = super::tiles::Tile> {
        super::tiles::TilesIterator::new(self.v, self.tile_size)
            .skip(self.shard_index)
            .step_by(self.shard_count)
    }

    #[cfg(not(target_os = "cuda"))]
    pub fn num_tiles(&self) -> u64 {
        num_integer::div_ceil(self.v, self.tile_size).pow(2) / (self.shard_count as u64)
    }

    pub fn pos_to_tile(&self, x: u64, y: u64) -> ((u64, u64), (u64, u64)) {
        let bx = x.div(self.tile_size) * self.tile_size;
        let by = y.div(self.tile_size) * self.tile_size;
        let ex = bx + self.tile_size;
        let ey = by + self.tile_size;
        ((bx, by), (ex, ey))
    }
}
