//! GPU Kernel for the graph generation.
#![cfg_attr(
    target_os = "cuda",
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![no_std]
#![allow(clippy::missing_safety_doc)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod kernels;
pub mod state;
