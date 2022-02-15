use criterion::{criterion_group, criterion_main, Criterion};

use girg_generator::benchmark::BenchmarkType;
use girg_generator::GeneratorMode;

fn criterion_benchmark(c: &mut Criterion) {
    girg_generator::benchmark::criterion_benchmark(c, GeneratorMode::GPU, BenchmarkType::TILE)
}

criterion_group!(gpu_size_scaling, criterion_benchmark);
criterion_main!(gpu_size_scaling);
