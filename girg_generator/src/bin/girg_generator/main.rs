use girg_generator::args::Args;
use girg_generator::{pbar, run_app};
use tracing::info;

fn main() -> anyhow::Result<()> {
    let app = Args::new_ref();
    pbar::setup_logging(None);

    if app.shard_index >= app.shard_count {
        panic!("Shard index must be less than shard count.");
    }

    if app.dimensions < 1 {
        panic!("Number of dimensions must be at least 1.");
    }

    info!("Running using the {:?} generator!", app.generator);

    run_app(app)
}
