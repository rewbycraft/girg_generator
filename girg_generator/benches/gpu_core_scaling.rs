use criterion::{Criterion, criterion_group, criterion_main};

use girg_generator::benchmark::BenchmarkType;
use girg_generator::GeneratorMode;

fn criterion_benchmark(c: &mut Criterion) {
    girg_generator::benchmark::criterion_benchmark(c, GeneratorMode::GPU, BenchmarkType::CORE)
}

criterion_group!(gpu_core_scaling, criterion_benchmark);
criterion_main!(gpu_core_scaling);