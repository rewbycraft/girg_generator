use std::time::Duration;

use criterion::{
    AxisScale, BenchmarkId, black_box, Criterion, PlotConfiguration, SamplingMode, Throughput,
};
use once_cell::sync::Lazy;
use anyhow::Context;

use crate::{Args, GeneratorMode, pbar, RandomMode, run_app};

fn get_suggested_launch_configuration() -> anyhow::Result<(u32, u32)> {
    let _ctx = cust::context::Context::create_and_push(
        cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO,
        (*CUST_DEVICE).clone(),
    )
        .context("create cuda context");
    let result = generator_gpu::suggested_launch_configuration().context("get_suggested_launch_config")?;
    Ok(result)
}

fn run(
    generator: GeneratorMode,
    workers: usize,
    tile_size: u64,
    vertices: u64,
    pregen: bool,
    device: Option<cust::device::Device>,
    blocks: Option<u32>,
) {
    let ctx = device.map(|device| {
        cust::context::Context::create_and_push(
            cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO,
            device,
        )
            .expect("create cuda context")
    });

    let args = Args {
        generator,
        random_mode: match pregen {
            true => RandomMode::PreGenerate,
            false => RandomMode::OnDemand,
        },
        workers,
        blocks,
        tile_size,
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
    TILE,
}

const SAMPLE_SIZE: usize = 10;
const GPU_THREADS: usize = 1;

static CUST_DEVICE: Lazy<cust::device::Device> = Lazy::new(|| {
    cust::init(cust::CudaFlags::empty()).expect("cuda init");

    let device_id = std::env::var("CUDA_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<u32>()
        .expect("parse device index");

    let device = cust::device::Device::get_device(device_id).expect("get device");

    println!(
        "Selected device {}: {} ({} MB)",
        device_id,
        device.name().expect("device name"),
        device.total_memory().expect("device total memory") / 1_000_000
    );

    device
});

pub fn criterion_benchmark(c: &mut Criterion, mode: GeneratorMode, ty: BenchmarkType) {
    pbar::setup_logging(Some("error".to_string()));

    let dev: Option<cust::device::Device> = if mode == GeneratorMode::GPU {
        Some((*CUST_DEVICE).clone())
    } else {
        None
    };

    let mode_str = match mode {
        GeneratorMode::CPU => "cpu",
        GeneratorMode::GPU => "gpu",
    };
    let type_str = match ty {
        BenchmarkType::SIZE => "size",
        BenchmarkType::CORE => "core",
        BenchmarkType::TILE => "tile",
    };
    let pgo: &[bool] = match ty {
        BenchmarkType::SIZE => &[true, false],
        BenchmarkType::CORE => &[true],
        BenchmarkType::TILE => &[true],
    };

    for &pregen in pgo {
        let pregen_str = match pregen {
            true => "pregenerate",
            false => "ondemand",
        };
        let group_name = format!("{}_{}_scaling_{}", mode_str, type_str, pregen_str);
        let mut group = c.benchmark_group(group_name);

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        group.plot_config(plot_config);
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(SAMPLE_SIZE);
        group.warm_up_time(Duration::from_secs(10 as u64));
        group.measurement_time(Duration::from_secs(60 as u64));

        match ty {
            BenchmarkType::SIZE => {
                let num_workers = match mode {
                    GeneratorMode::CPU => num_cpus::get(),
                    GeneratorMode::GPU => GPU_THREADS,
                };
                let range_end = match mode {
                    GeneratorMode::CPU => 18,
                    GeneratorMode::GPU => 20,
                };

                for vertices in (10..=range_end).map(|exp| 2u64.pow(exp)) {
                    group.throughput(Throughput::Elements(vertices));
                    group.bench_with_input(
                        BenchmarkId::from_parameter(&vertices),
                        &vertices,
                        |b, &vertices| {
                            b.iter(|| run(mode, num_workers, 1000, vertices, pregen, dev.clone(), None));
                        },
                    );
                }
            }

            BenchmarkType::CORE => match mode {
                GeneratorMode::CPU => {
                    let num_cores = num_cpus::get();
                    for cpus in 1..=num_cores {
                        group.throughput(Throughput::Elements(cpus as u64));
                        group.bench_with_input(
                            BenchmarkId::from_parameter(&cpus),
                            &cpus,
                            |b, &cpus| {
                                b.iter(|| run(mode, cpus, 1000, 100_000, pregen, None, None));
                            },
                        );
                    }
                }
                GeneratorMode::GPU => {
                    let (grid_size, block_size) = get_suggested_launch_configuration().expect("get_suggested_launch_configuration");
                    let max_val = (grid_size + (grid_size / 2)) / 2;
                    for blocks in (1..=max_val).map(|e| e * 2) {
                        group.throughput(Throughput::Elements((blocks as u64) * (block_size as u64)));
                        group.bench_with_input(
                            BenchmarkId::from_parameter(&blocks),
                            &blocks,
                            |b, &blocks| {
                                b.iter(|| {
                                    run(mode, GPU_THREADS, 1000, 500_000, pregen, dev.clone(), Some(blocks))
                                });
                            },
                        );
                    }
                }
            },

            BenchmarkType::TILE => {
                let num_workers = match mode {
                    GeneratorMode::CPU => num_cpus::get(),
                    GeneratorMode::GPU => GPU_THREADS,
                };
                let range_end = match mode {
                    GeneratorMode::CPU => 16,
                    GeneratorMode::GPU => 13,
                };
                let vertices = match mode {
                    GeneratorMode::CPU => 65536,
                    GeneratorMode::GPU => 262144,
                };

                for tile_size in (7..=range_end).map(|exp| 2u64.pow(exp)) {
                    group.throughput(Throughput::Elements(tile_size));
                    group.bench_with_input(
                        BenchmarkId::from_parameter(&tile_size),
                        &tile_size,
                        |b, &tile_size| {
                            b.iter(|| run(mode, num_workers, tile_size, vertices, pregen, dev.clone(), None));
                        },
                    );
                }
            },
        }
        group.finish();
    }
}
