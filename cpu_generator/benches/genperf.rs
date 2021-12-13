use criterion::{BenchmarkId, black_box, Criterion, criterion_group, criterion_main};
use cpu_generator::GenerationParameters;

trait WorkFun {
    fn worker_function<F: FnMut(u64, u64)>(&self, start: (u64, u64), end: (u64, u64), params: &cpu_generator::GenerationParameters, cb: F);
}

struct WorkFun1 {
}

impl WorkFun for WorkFun1 {
    fn worker_function<F: FnMut(u64, u64)>(&self, start: (u64, u64), end: (u64, u64), params: &GenerationParameters, cb: F) {
        cpu_generator::worker_function(start, end, params, cb)
    }
}

struct WorkFun2 {
}

impl WorkFun for WorkFun2 {
    fn worker_function<F: FnMut(u64, u64)>(&self, start: (u64, u64), end: (u64, u64), params: &GenerationParameters, cb: F) {
        cpu_generator::worker_function_pregen(start, end, params, cb)
    }
}

fn worker_function_bench<T: WorkFun>(v: u64, wf: T) -> usize
{
    let pareto = cpu_generator::random::ParetoDistribution::new(1., 1.5);
    let params = cpu_generator::GenerationParameters::new(pareto, 1., v);

    let mut edges = black_box(0usize);
    wf.worker_function(black_box((0, 0)), black_box((v, v)), black_box(&params), |_i, _j| {
        edges += 1;
    });
    edges
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_function");
    for v in [100u64, 500u64, 1000u64, 2000u64] {
        group.bench_with_input(BenchmarkId::new("compute", v), &v, |b, v| b.iter(|| {
            let v = *v;
            worker_function_bench(v, WorkFun1{})
        }));
        group.bench_with_input(BenchmarkId::new("pregen", v), &v, |b, v| b.iter(|| {
            let v = *v;
            worker_function_bench(v, WorkFun2{})
        }));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
