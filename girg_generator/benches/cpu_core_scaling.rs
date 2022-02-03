use criterion::{criterion_group, criterion_main, Criterion};

use girg_generator::benchmark::BenchmarkType;
use girg_generator::GeneratorMode;

fn criterion_benchmark(c: &mut Criterion) {
    girg_generator::benchmark::criterion_benchmark(c, GeneratorMode::CPU, BenchmarkType::CORE)
}

criterion_group!(cpu_core_scaling, criterion_benchmark);
criterion_main!(cpu_core_scaling);
