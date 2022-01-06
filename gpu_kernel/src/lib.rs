#![cfg_attr(
target_os = "cuda",
no_std,
feature(register_attr),
register_attr(nvvm_internal)
)]
#![allow(clippy::missing_safety_doc)]

#[cfg(target_os = "cuda")]
pub mod kernels;
pub mod state;
