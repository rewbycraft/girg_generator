use crate::params::VecSeeds;
use crate::tiles::Tile;
use crossbeam_channel::{Receiver, Sender};
use generator_core::params::GenerationParameters;

pub type EdgeSender = Sender<Vec<(u64, u64)>>;

pub trait GraphGenerator: Sized {
    type ConstructArgument: Clone + Send + 'static;

    fn new(arg: Self::ConstructArgument) -> anyhow::Result<Self>;

    fn generate(
        &self,
        output_sender: EdgeSender,
        finished_job_sender: Sender<Tile>,
        new_job_receiver: Receiver<Tile>,
        params: &GenerationParameters<VecSeeds>,
    ) -> anyhow::Result<()>;
}
