use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    //Make cargo re-run this if the common crate changed.
    println!("cargo:rerun-if-changed=../generator_common");
    let mut builder = CudaBuilder::new("../gpu_kernel");
    //Specify nvidia build arch.
    builder.arch = NvvmArch::Compute50;

    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("kernel.ptx");
    let output = builder
        .copy_to(dest_path)
        .build()
        .unwrap();
    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", output.as_path().display());
}
