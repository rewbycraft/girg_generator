//! This is the main application crate.
//!
//! Here lives the code that ties the whole application together and acts as a coordinator of all the different subcomponents.
//!
//! The [generator/core](generator_core) crate provides the core of the generation logic.
//! It contains the main probability function as well as the struct that holds the parameters for it.
//! This crate is used both on the CPU and on the GPU directly. As such, it is kept minimal and is a `no_std` crate.
//! Generally this is where you should look to implement new probability functions.
//!
//! The [generator/common](generator_common) crate then extends [generator/core](generator_core) with common behaviour.
//! Amongst this behaviour is a [function](generator_common::algorithm::generate_parameters) one can amend to pre-calculate properties like the sum of all weights.
//!
//! The [generator/cpu](generator_cpu) and [generator/gpu](generator_gpu) crates provide two implementations of the GIRG generator.
//!
//!

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use std::fs::File;
use std::io::prelude::*;

use tracing::{debug, info};

use crate::args::{ArgsRef, GeneratorMode};
use crate::parquet_edges::ParquetEdgeWriter;

pub mod args;
pub mod parquet_edges;
pub mod pbar;
#[cfg(test)]
pub mod tests;

/// Main function of the application.
///
/// This functions is the main entrypoint for the application after the arguments have been parsed and logging has been initialized.
pub fn run_app(app: ArgsRef) -> anyhow::Result<()> {
    info!("Get params...");
    let params = app.get_params();

    info!("Params:\n{:#?}", params);

    if let Some(p) = app.output_weights.as_ref() {
        info!("Writing weights file...");
        let mut f = File::create(p).expect("Unable to create file");
        for i in params.compute_weights() {
            writeln!(f, "{}", i).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_positions.as_ref() {
        info!("Writing positions file...");
        let mut f = File::create(p).expect("Unable to create file");
        for i in params.compute_positions() {
            for j in 0..i.len() {
                write!(f, "{}", i[j]).unwrap();
                if j < i.len() - 1 {
                    write!(f, ",").unwrap();
                }
            }
            writeln!(f).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    let (tile_sender, tile_receiver) = crossbeam_channel::bounded(5);
    let (edge_sender, edge_receiver) = crossbeam_channel::bounded(100);
    let (finish_sender, finish_receiver) = crossbeam_channel::bounded(10000);

    pbar::create_progress_bar(params.num_tiles());

    let mut handles = match app.generator {
        #[cfg(feature = "gpu")]
        GeneratorMode::GPU => {
            generator_common::threads::start_workers::<generator_gpu::GPUGenerator>(
                app.clone(),
                app.workers,
                edge_sender,
                finish_sender,
                tile_receiver,
                &params,
            )
        }
        GeneratorMode::CPU => {
            generator_common::threads::start_workers::<generator_cpu::CPUGenerator>(
                (),
                app.workers,
                edge_sender,
                finish_sender,
                tile_receiver,
                &params,
            )
        }
    };
    handles.push(generator_common::threads::start_generate_tiles_thread(
        tile_sender,
        &params,
    ));

    let mut degree_counters: Vec<usize> = Vec::new();
    degree_counters.resize(app.vertices as usize, 0usize);

    handles.push(std::thread::spawn(move || {
        pbar::increment_progress(0);
        for b in finish_receiver {
            debug!("Finished block {:?}.", b);
            pbar::increment_progress(1);
        }
    }));

    {
        info!("Receiving edges...");
        let mut edge_counter = 0u128;

        let mut csv_wtr = app.output_edges_csv.as_ref().map(|p| {
            let mut wtr = csv::Writer::from_path(p).unwrap();
            wtr.write_record(&["edge_i", "edge_j"]).unwrap();
            wtr
        });

        let mut parquet_wtr = app
            .output_edges_parquet
            .as_ref()
            .map(ParquetEdgeWriter::new);

        for edge_tile in edge_receiver {
            if let Some(wtr) = parquet_wtr.as_mut() {
                wtr.write_vec(&edge_tile);
            }

            for (i, j) in edge_tile {
                edge_counter += 1;
                *degree_counters.get_mut(i as usize).unwrap() += 1;
                if let Some(wtr) = csv_wtr.as_mut() {
                    wtr.write_record(&[format!("{}", i), format!("{}", j)])
                        .unwrap();
                }
            }
        }

        if let Some(wtr) = csv_wtr.as_mut() {
            wtr.flush().unwrap();
        }

        if let Some(wtr) = parquet_wtr.as_mut() {
            wtr.close();
        }

        info!("All edges received! ({} edges)", edge_counter);
    }

    info!("Waiting for the threads to join...");
    while let Some(h) = handles.pop() {
        h.join().unwrap();
    }
    info!("Threads joined!");

    pbar::finish_progress_bar();

    //info!("Degrees: {:?}", degree_counters);

    if let Some(p) = app.output_degrees_csv.as_ref() {
        info!("Writing degree csv...");
        let mut wtr = csv::Writer::from_path(p).unwrap();
        wtr.write_record(&["node_id", "degree"]).unwrap();
        for (i, j) in degree_counters.iter().enumerate() {
            wtr.write_record(&[format!("{}", i), format!("{}", *j)])
                .unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_degrees_txt.as_ref() {
        info!("Writing degree txt...");
        let mut f = File::create(p).expect("Unable to create file");
        for i in degree_counters.iter() {
            writeln!(f, "{}", i).unwrap();
        }
        f.flush().unwrap();
        info!("Done writing!");
    }

    if let Some(p) = app.output_degrees_distribution.as_ref() {
        info!("Writing degree distribution csv...");
        let mut wtr = csv::Writer::from_path(p).unwrap();
        wtr.write_record(&["x", "number of nodes with degree > x / number of nodes"])
            .unwrap();
        for x in 0..=degree_counters.len() {
            let s: f64 = degree_counters.iter().filter(|&d| *d > x as usize).count() as f64;
            let v = s / (degree_counters.len() as f64);
            let x = x as f64;
            wtr.write_record(&[format!("{}", x), format!("{}", v)])
                .unwrap();
        }
        wtr.flush().unwrap();
        info!("Done writing!");
    }

    Ok(())
}
