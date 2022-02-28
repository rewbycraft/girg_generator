pub use generator_core::algorithm::*;
use tracing::info;
use crate::params::CPUGenerationParameters;

pub fn generate_parameters(params: &mut CPUGenerationParameters) {
    info!("Computing W...");
    params.w = params.compute_weights().into_iter().sum();
    info!("Computed W = {}", params.w);
}
