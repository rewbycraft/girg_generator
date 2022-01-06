use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    let mut builder = CudaBuilder::new("../gpu_kernel");
    builder.arch = NvvmArch::Compute50;
    println!("cargo:rerun-if-changed=../generator_common");
    builder
        .copy_to("./kernel.ptx")
        .build()
        .unwrap();
}
