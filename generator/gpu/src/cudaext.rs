use anyhow::Context;
use cust::memory::DeviceBuffer;
use cust::stream::Stream;

use generator_common::params::{GenerationParameters, RawSeeds, VecSeeds};

pub trait GenerationParametersCudaExt {
    unsafe fn as_rawseeds_async(
        &self,
        stream: &Stream,
    ) -> anyhow::Result<(GenerationParameters<RawSeeds>, DeviceBuffer<u64>)>;
}

impl GenerationParametersCudaExt for GenerationParameters<VecSeeds> {
    unsafe fn as_rawseeds_async(
        &self,
        stream: &Stream,
    ) -> anyhow::Result<(GenerationParameters<RawSeeds>, DeviceBuffer<u64>)> {
        let buffer = self
            .seeds
            .get_dbuffer_async(stream)
            .context("get_dbuffer_async")?;
        let params = GenerationParameters {
            seeds: RawSeeds::new(buffer.as_device_ptr().as_ptr()),
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
        };

        Ok((params, buffer))
    }
}

pub trait VecSeedsCudaExt {
    fn get_dbuffer(&self) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>>;
    unsafe fn get_dbuffer_async(
        &self,
        stream: &cust::stream::Stream,
    ) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>>;
}

impl VecSeedsCudaExt for VecSeeds {
    fn get_dbuffer(&self) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>> {
        cust::memory::DeviceBuffer::from_slice(&self.seeds)
    }

    unsafe fn get_dbuffer_async(
        &self,
        stream: &cust::stream::Stream,
    ) -> cust::error::CudaResult<cust::memory::DeviceBuffer<u64>> {
        cust::memory::DeviceBuffer::from_slice_async(&self.seeds, stream)
    }
}
