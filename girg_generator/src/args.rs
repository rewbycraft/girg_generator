use clap::{ArgEnum, Parser, ValueHint};
use generator_common::params::{CPUGenerationParameters, GenerationParameters};
use generator_common::random::ParetoDistribution;
use std::path::PathBuf;
use std::sync::Arc;
use strum::EnumIter;

pub type ArgsRef = Arc<Args>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug, EnumIter)]
pub enum GeneratorMode {
    CPU,
    #[cfg(feature = "gpu")]
    GPU,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum, Debug)]
pub enum RandomMode {
    PreGenerate,
    OnDemand,
}

/// GIRG Generator
#[derive(Parser, Debug)]
pub struct Args {
    /// What generator to use
    #[clap(short, long, arg_enum)]
    pub generator: GeneratorMode,
    /// How to use the randomness
    #[clap(long, arg_enum, default_value_t = RandomMode::PreGenerate)]
    pub random_mode: RandomMode,
    /// Number of worker threads
    #[clap(short, long, default_value_t = 1)]
    pub workers: usize,
    /// Number of vertices
    #[clap(long, default_value_t = 1000)]
    pub tile_size: u64,
    /// Number of vertices
    #[clap(short, long, default_value_t = 1_000_000)]
    pub vertices: u64,
    /// Alpha value of the probability function
    #[clap(short, long, default_value_t = 1.5)]
    pub alpha: f32,
    /// Alpha value of the pareto distribution
    #[clap(short, long, default_value_t = 1.5)]
    pub beta: f32,
    /// x_min value of the pareto distribution
    #[clap(short, long, default_value_t = 1.0)]
    pub x_min: f32,
    /// Number of spatial dimensions
    #[clap(short, long, default_value_t = 2)]
    pub dimensions: usize,
    /// Number of shards in use.
    #[clap(long, default_value_t = 1)]
    pub shard_count: usize,
    /// Index of this shard. [0,shard_count)
    #[clap(long, default_value_t = 0)]
    pub shard_index: usize,
    /// NVidia Device index
    #[clap(long, default_value_t = 0)]
    pub device: u32,
    /// Override the amount of blocks used by the gpu.
    #[clap(long)]
    pub blocks: Option<u32>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees_distribution to
    pub output_degrees_distribution: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (csv format: node_id, degree)
    pub output_degrees_csv: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write degrees to (plain text: degree\n)
    pub output_degrees_txt: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write edges to (csv: i, j)
    pub output_edges_csv: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write edges to (parquet: i, j) (recommended due to compression)
    pub output_edges_parquet: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write weights to (plain text, one weight per line)
    pub output_weights: Option<PathBuf>,
    #[clap(long, parse(from_os_str), value_hint = ValueHint::FilePath)]
    /// File to write position to (csv: one column per dimension)
    pub output_positions: Option<PathBuf>,
    /// Seed values
    #[clap(long, short)]
    pub seeds: Option<Vec<u64>>,
    /// Size of the buffer used to hold edges before processing. Effectively edge batch size.
    #[clap(long, default_value_t = 1024)]
    pub edgebuffer_size: u64,
}

impl Args {
    pub fn new_ref() -> ArgsRef {
        Arc::new(Self::parse())
    }

    pub fn get_pareto(&self) -> ParetoDistribution {
        ParetoDistribution::new(self.x_min, self.beta)
    }

    pub fn get_params(&self) -> CPUGenerationParameters {
        match self.seeds.as_ref() {
            None => CPUGenerationParameters::new(
                self.dimensions,
                self.get_pareto(),
                self.alpha,
                self.vertices,
                self.tile_size,
                self.edgebuffer_size,
                self.random_mode == RandomMode::PreGenerate,
                self.blocks.unwrap_or(0),
                self.shard_index,
                self.shard_count,
            ),
            Some(s) => CPUGenerationParameters::from_seeds(
                self.dimensions,
                self.get_pareto(),
                self.alpha,
                self.vertices,
                s.as_slice(),
                self.tile_size,
                self.edgebuffer_size,
                self.random_mode == RandomMode::PreGenerate,
                self.blocks.unwrap_or(0),
                self.shard_index,
                self.shard_count,
            ),
        }
    }
}

#[cfg(feature = "gpu")]
impl generator_gpu::GPUGeneratorArguments for Args {
    fn get_device(&self) -> u32 {
        self.device
    }
}
