pub use generator_core::params::*;

pub mod ext;

#[derive(Debug, Clone)]
pub struct VecSeeds {
    pub seeds: Vec<u64>,
}

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

