//! Definition of a graph generator.

use crate::params::CPUGenerationParameters;
use crate::tiles::Tile;
use crossbeam_channel::{Receiver, Sender};
use generator_core::tiles::Edge;

/// Sender that is used to submit edge buffers.
pub type EdgeSender = Sender<Vec<Edge>>;

/// Trait defining the operations of a graph generator.
///
/// This allows the actual generator to be pluggable.
pub trait GraphGenerator: Sized {
    /// The type of argument taken to construct a new one of these.
    type ConstructArgument: Clone + Send + 'static;

    /// Create a new instance.
    fn new(arg: Self::ConstructArgument) -> anyhow::Result<Self>;

    /// Generate a graph.
    fn generate(
        &self,
        output_sender: EdgeSender,
        finished_job_sender: Sender<Tile>,
        new_job_receiver: Receiver<Tile>,
        params: &CPUGenerationParameters,
    ) -> anyhow::Result<()>;
}
