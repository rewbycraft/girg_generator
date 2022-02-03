#[cfg(not(target_os = "cuda"))]
pub mod cpu;

pub mod gpu;
