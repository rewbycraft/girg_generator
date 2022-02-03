// use std::sync::Once;
//
// use mur3::murmurhash3_x64_128;
// use rstest::*;
// use tracing::{debug, info, instrument};
//
// use common::params::{DIMENSIONS, GenerationParameters};
// use common::random::ParetoDistribution;
//
// use crate::GeneratorMode;
//
// static INIT: Once = Once::new();
// static INIT_GPU: Once = Once::new();
//
// fn init() {
//     INIT.call_once(|| {
//         std::env::set_var(
//             "RUST_LOG",
//             std::env::var("RUST_LOG").unwrap_or("info".to_string()),
//         );
//         tracing_subscriber::fmt().with_test_writer().init();
//     });
// }
//
// fn cuda_init() {
//     INIT_GPU.call_once(|| {
//         info!("CUDA init...");
//         cust::init(cust::CudaFlags::empty()).expect("init cust");
//     });
// }
//
// #[instrument(skip(params))]
// fn compute_edge_hash(generator: GeneratorMode, params: &GenerationParameters) -> u128 {
//     init();
//
//     info!("Generating...");
//
//     let ctx: Option<cust::context::Context> = if generator == GeneratorMode::GPU {
//         cuda_init();
//
//         info!("CUDA device get...");
//         let device = cust::device::Device::get_device(0).expect("cust_get_device");
//
//         info!("Device: {} ({} MB)", device.name().expect("device_name"), device.total_memory().expect("device_memory") / 1_000_000);
//
//         Some(cust::context::Context::create_and_push(cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO, device).expect("create_context"))
//     } else {
//         None
//     };
//
//
//     let (tile_sender, tile_receiver) = crossbeam_channel::bounded(5);
//     let (edge_sender, edge_receiver) = crossbeam_channel::bounded(5);
//     let (finish_sender, finish_receiver) = crossbeam_channel::bounded(10000);
//
//     let mut handles = match generator {
//         GeneratorMode::GPU => common::threads::start_workers::<gpu::GPUGenerator>(&ctx, 1, edge_sender, finish_sender, tile_receiver, &params),
//         GeneratorMode::CPU => common::threads::start_workers::<cpu::CPUGenerator>(&ctx, 4, edge_sender, finish_sender, tile_receiver, &params),
//     };
//     handles.push(common::threads::start_generate_tiles_thread(tile_sender, &params));
//
//
//     handles.push(std::thread::spawn(move || {
//         for b in finish_receiver {
//             debug!("Finished block {:?}.", b);
//         }
//     }));
//
//     let mut sum = 0u128;
//     let mut count = 0usize;
//     info!("Receiving edges...");
//     for edge_tile in edge_receiver {
//         for (i, j) in edge_tile {
//             let (h, _) = murmurhash3_x64_128(&[i.to_be_bytes(), j.to_be_bytes()].concat(), 0);
//             let h = h as u128;
//             sum += h;
//             count += 1;
//         }
//     }
//     info!("All edges received!");
//
//     info!("Waiting for the threads to join...");
//     while let Some(h) = handles.pop() {
//         h.join().unwrap();
//     }
//     info!("Threads joined!");
//
//     info!("Number of edges: {}", count);
//     info!("Computed hash: {}", sum);
//
//     sum
// }
//
// #[rstest]
// #[trace]
// // #[case(10000, 1.5f32, 1.0f32, 1.5f32, 1000)]
// // #[case(10000, 1.5f32, 1.0f32, 2.5f32, 1000)]
// // #[case(10000, 1.5f32, 2.0f32, 1.5f32, 1000)]
// // #[case(10000, 2.5f32, 1.0f32, 1.5f32, 1000)]
// // #[case(10000, f32::INFINITY, 1.0f32, 1.5f32, 1000)]
// // #[case(100000, 1.5f32, 1.0f32, 1.5f32, 1000)]
// // #[case(100000, f32::INFINITY, 1.0f32, 1.5f32, 1000)]
// fn test_cpu_gpu_equality(
//     #[values(10_000, 20_000)] v: u64,
//     #[values(1.0f32, 1.1f32, 1.5f32, 2.5f32)] alpha: f32,
//     #[values(1.0f32, 1.1f32, 1.5f32, 2.5f32)] x_min: f32,
//     #[values(1.0f32, 1.1f32, 1.5f32, 2.5f32)] beta: f32,
//     #[values(1000)] tile_size: u64,
// ) {
//     init();
//     let pareto = ParetoDistribution::new(x_min, beta);
//     let params = GenerationParameters::new(pareto, alpha, v, tile_size);
//     info!("Params:\n{:#?}", params);
//     let cpu_hash = compute_edge_hash(GeneratorMode::CPU, &params);
//     let gpu_hash = compute_edge_hash(GeneratorMode::GPU, &params);
//     assert_eq!(gpu_hash, cpu_hash, "expected equal hashes between cpu and gpu");
// }
//
// #[rstest]
// #[trace]
// #[case(10000, 1.1, 1.0, 2.5, 1000, [3702171088734132669, 7758113088146926290, 9158248949434531752, 12627271752717934084])]
// #[case(20000, 1.5, 2.5, 1.1, 1000, [17347240448019473911, 6482979354306528467, 6393418829833712867, 1917765620565624317])]
// #[case(20000, 1.1, 2.5, 1.0, 1000, [2462949633862107540, 3870694928585443708, 18190401166422532476, 17746881630084208051])]
// #[case(20000, 1.1, 1.1, 1.5, 1000, [16146293082247982091, 3074503608420495527, 13221120962825523065, 10135440224367571005])]
// #[case(20000, 1.0, 2.5, 2.5, 1000, [641528916818287623, 14583863194921507506, 2172636905578962766, 9292436636520362153])]
// #[case(10000, 1.5, 2.5, 1.1, 1000, [13498820208893652259, 10871832395042491586, 1215117311171283993, 14549546664141665766])]
// #[case(10000, 1.1, 1.5, 1.0, 1000, [8860793245296651914, 4875508041214000327, 8043432004899630584, 14121560416875303279])]
// #[case(10000, 1.0, 1.0, 1.0, 1000, [9943627937936294394, 17623916284063759097, 9773449833268882578, 13586026909810947487])]
// fn test_cpu_gpu_equality_specific(
//     #[case] v: u64,
//     #[case] alpha: f32,
//     #[case] x_min: f32,
//     #[case] beta: f32,
//     #[case] tile_size: u64,
//     #[case] seeds: [u64; DIMENSIONS + 2],
// ) {
//     init();
//     let pareto = ParetoDistribution::new(x_min, beta);
//     let params = GenerationParameters::from_seeds(pareto, alpha, v, seeds, tile_size);
//     info!("Params:\n{:#?}", params);
//     let cpu_hash = compute_edge_hash(GeneratorMode::CPU, &params);
//     let gpu_hash = compute_edge_hash(GeneratorMode::GPU, &params);
//     // let v = cpu_set.difference(&gpu_set).cloned().collect::<Vec<(u64, u64)>>();
//     // for (i, j) in v.iter().copied() {
//     //     let w_i = params.compute_weight(i);
//     //     let w_j = params.compute_weight(j);
//     //     let p_i = params.compute_position(i);
//     //     let p_j = params.compute_position(j);
//     //     info!("i: {} (w = {}), (p = {:?})", i, w_i, p_i);
//     //     info!("j: {} (w = {}), (p = {:?})", j, w_j, p_j);
//     //     let d = compute_distance(p_i, p_j);
//     //     info!("d: {}", d);
//     //     let p = compute_probability(d, w_i, w_j, &params);
//     //     info!("p: {}", p);
//     //     let rp = common::random::random_edge(i, j, params.get_seed(SeedEnum::Edge));
//     //     info!("rp: {}", rp);
//     //     let diff = p - rp;
//     //     info!("diff: {}", diff);
//     // }
//     // assert_eq!(v, vec![], "expected no difference");
//     assert_eq!(gpu_hash, cpu_hash, "expected equal hashes between cpu and gpu");
// }
//
