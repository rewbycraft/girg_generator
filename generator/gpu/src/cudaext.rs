//! GPU version of the [`GenerationParameters`] + an extension to [`CPUGenerationParameters`].

use std::ops::Deref;
use cust::stream::Stream;
use fixed_size_buffer::GetFixedSizeBufferRef;
use fixed_size_buffer::gpu::{AsGPUBuffer, FixedSizeGPUBuffer};

use generator_common::params::{CPUGenerationParameters, GenerationParameters};

/// Instance of [`GenerationParameters`] with the seeds stored on the GPU.
pub struct GPUGenerationParameters<'a> {
    /// Parameters.
    params: GenerationParameters<'a>,
    /// Backing storage for the seeds buffer.
    #[allow(dead_code)]
    real_seeds: FixedSizeGPUBuffer<'a, u64>,
}

/// Extension to the [`CPUGenerationParameters`] to allow conversion to [`GPUGenerationParameters`].
pub trait GenerationParametersCudaExt {
    /// Derive a [`GPUGenerationParameters`] from this.
    unsafe fn as_gpu_params(&self, stream: &Stream) -> anyhow::Result<GPUGenerationParameters>;
}

impl GenerationParametersCudaExt for CPUGenerationParameters<'_> {
    unsafe fn as_gpu_params(&self, stream: &Stream) -> anyhow::Result<GPUGenerationParameters> {
        let real_seeds = self.real_seeds.as_gpu_buffer_async(stream)?;

        let params = self.params.replace_seeds(real_seeds.get_ref());
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
