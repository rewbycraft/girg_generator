use std::time::Duration;

use criterion::{AxisScale, BenchmarkId, black_box, Criterion, PlotConfiguration, SamplingMode, Throughput};

use crate::{Args, GeneratorMode, pbar, RandomMode, run_app};

fn run(generator: GeneratorMode, workers: usize, vertices: u64, pregen: bool, device: Option<cust::device::Device>, blocks: Option<u32>) {
    let ctx = device.map(|device| {
        cust::context::Context::create_and_push(cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO, device).expect("create cuda context")
    });

    let args = Args {
        generator,
        random_mode: match pregen {
            true => RandomMode::PreGenerate,
            false => RandomMode::OnDemand,
        },
        workers,
        blocks,
        tile_size: 1000,
        vertices,
        alpha: 1.5,
        beta: 1.5,
        x_min: 1.0,
        dimensions: 2,
        shard_count: 1,
        shard_index: 0,
        device: 0,
        output_degrees_distribution: None,
        output_degrees_csv: None,
        output_degrees_txt: None,
        output_edges_csv: None,
        output_edges_parquet: None,
        output_weights: None,
        output_positions: None,
        seeds: None,
        edgebuffer_size: 10240,
    };

    run_app(black_box(args), black_box(ctx)).unwrap();
}

pub enum BenchmarkType {
    SIZE,
    CORE,
}

const SAMPLE_SIZE: usize = 10;

pub fn criterion_benchmark(c: &mut Criterion, mode: GeneratorMode, ty: BenchmarkType) {
    pbar::setup_logging(Some("error".to_string()));

    let dev: Option<cust::device::Device> = if mode == GeneratorMode::GPU {
        cust::init(cust::CudaFlags::empty()).expect("cuda init");
        let device = cust::device::Device::get_device(0).expect("get device");

        println!("Device: {} ({} MB)", device.name().expect("device name"), device.total_memory().expect("device total memory") / 1_000_000);

        Some(device)
    } else {
        None
    };

    for pregen in [true, false] {
        let mode_str = match mode {
            GeneratorMode::CPU => "cpu",
            GeneratorMode::GPU => "gpu",
        };
        let pregen_str = match pregen {
            true => "pregen",
            false => "ondemand",
        };
        let type_str = match ty {
            BenchmarkType::SIZE => "size",
            BenchmarkType::CORE => "core",
        };
        let group_name = format!("{}_{}_scaling_{}", mode_str, type_str, pregen_str);
        let mut group = c.benchmark_group(group_name);


        let plot_config = PlotConfiguration::default()
            .summary_scale(AxisScale::Logarithmic);
        group.plot_config(plot_config);
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(SAMPLE_SIZE);
        group.warm_up_time(Duration::from_secs(10 as u64));
        group.measurement_time(Duration::from_secs(60 as u64));

        match ty {
            BenchmarkType::SIZE => {
                let num_workers = match mode {
                    GeneratorMode::CPU => num_cpus::get(),
                    GeneratorMode::GPU => 3,
                };
                let range_end = match mode {
                    GeneratorMode::CPU => 10,
                    GeneratorMode::GPU => 24,
                };

                for vertices in (0..=range_end).map(|exp| 1_024u64 * (2u64.pow(exp))) {
                    group.throughput(Throughput::Elements(vertices));
                    group.bench_with_input(BenchmarkId::from_parameter(&vertices), &vertices, |b, &vertices| {
                        b.iter(|| run(mode, num_workers, vertices, pregen, dev.clone(), None));
                    });
                }
            }

            BenchmarkType::CORE => {
                match mode {
                    GeneratorMode::CPU => {
                        let num_cores = num_cpus::get();
                        for cpus in 1..num_cores {
                            group.throughput(Throughput::Elements(cpus as u64));
                            group.bench_with_input(BenchmarkId::from_parameter(&cpus), &cpus, |b, &cpus| {
                                b.iter(|| run(mode, cpus, 100_000, pregen, None, None));
                            });
                        }
                    }
                    GeneratorMode::GPU => {
                        for blocks in (2..=7).map(|e| 2u32.pow(e)) {
                            group.throughput(Throughput::Elements(blocks as u64));
                            group.bench_with_input(BenchmarkId::from_parameter(&blocks), &blocks, |b, &blocks| {
                                b.iter(|| run(mode, 3, 1_000_000, pregen, dev.clone(), Some(blocks)));
                            });
                        }
                    }
                }
            }
        }
        group.finish();
    }
}
