use crate::params::ext::GenerationParametersExt;
use crate::params::VecSeeds;
pub use generator_core::algorithm::*;
use generator_core::params::GenerationParameters;
use tracing::info;

pub fn generate_parameters(params: &mut GenerationParameters<VecSeeds>) {
    info!("Computing W...");
    params.w = params.compute_weights().into_iter().sum();
    info!("Computed W = {}", params.w);
}
