//! Common functions for generators.
//!
//! This crate contains the shared functions and types that are not able to be put in a `no_std` environment.
//! And thus are unable to be present in the [`generator_core`] crate.
//! This crate also re-exports all of [`generator_core`] for convenience.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod algorithm;
pub mod generator;
pub mod params;
pub mod random;
pub mod threads;
pub mod tiles;

pub use generator_core::MAX_DIMS;
