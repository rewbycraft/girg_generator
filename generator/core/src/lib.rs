//! The core of the generation algorithm.
//!
//! This crate implements the core functionality for the generation algorithm.
//! Notably, this crate contains the probability function itself.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(
    target_os = "cuda",
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![no_std]
#![allow(clippy::missing_safety_doc)]

pub mod algorithm;
pub mod params;
pub mod random;
pub mod memory;

/// This tells you how many dimensions you can have at maximum when doing GPU computing using on-demand randomness.
pub const MAX_DIMS: usize = 2;
