//! Module providing some additional functions on top of [`generator_core::algorithm`].

pub use generator_core::algorithm::*;
use tracing::info;
use crate::params::CPUGenerationParameters;

/// Function one can implement to pre-generate certain properties of the graph.
pub fn generate_parameters(params: &mut CPUGenerationParameters) {
    info!("Computing W...");
    params.w = params.compute_weights().into_iter().sum();
    info!("Computed W = {}", params.w);
}
