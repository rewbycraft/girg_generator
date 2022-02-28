use std::ops::Deref;
use anyhow::Context;
use cust::memory::{AsyncCopyDestination, DeviceBuffer};
use cust::stream::Stream;
use fixed_size_buffer::GetFixedSizeBufferRef;
use fixed_size_buffer::gpu::{AsGPUBuffer, FixedSizeGPUBuffer};

use generator_common::params::{CPUGenerationParameters, GenerationParameters};

pub struct GPUGenerationParameters<'a> {
    params: GenerationParameters<'a>,
    real_seeds: FixedSizeGPUBuffer<'a, u64>,
}

pub trait GenerationParametersCudaExt {
    unsafe fn as_gpu_params(&self, stream: &Stream) -> anyhow::Result<GPUGenerationParameters>;
}

impl GenerationParametersCudaExt for CPUGenerationParameters<'_> {
    unsafe fn as_gpu_params(&self, stream: &Stream) -> anyhow::Result<GPUGenerationParameters> {
        let real_seeds = self.real_seeds.as_gpu_buffer_async(stream)?;

        let params = self.params.clone().replace_seeds(real_seeds.get_ref());
        Ok(GPUGenerationParameters {
            params,
            real_seeds,
        })
    }
}

impl<'a> Deref for GPUGenerationParameters<'a> {
    type Target = GenerationParameters<'a>;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}
