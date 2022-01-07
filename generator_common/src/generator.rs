use crossbeam_channel::{Sender, Receiver};
use crate::params::{VecSeeds, GenerationParameters};

pub trait GraphGenerator: Sized {
    fn new() -> anyhow::Result<Self>;
    fn generate(&self, sender: Sender<Vec<(u64, u64)>>, finisher: Sender<((u64, u64), (u64, u64))>, receiver: Receiver<((u64, u64), (u64, u64))>, params: &GenerationParameters<VecSeeds>) -> anyhow::Result<()>;
}
