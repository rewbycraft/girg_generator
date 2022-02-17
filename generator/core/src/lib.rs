#![cfg_attr(target_os = "cuda", feature(register_attr), register_attr(nvvm_internal))]
#![no_std]
#![allow(clippy::missing_safety_doc)]

pub mod algorithm;
pub mod params;
pub mod random;

// This tells you how many dimensions you can have at maximum when doing GPU computing using on-demand.
pub const MAX_DIMS: usize = 2;