use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    eprintln!(
        "AAA {}",
        std::env::var_os("LD_LIBRARY_PATH")
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
    );
    eprintln!(
        "BBB {}",
        std::env::var_os("CARGO_PKG_NAME")
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
    );
    eprintln!(
        "CCC {}",
        std::env::var_os("TARGET")
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
    );
    eprintln!(
        "DDD {}",
        std::env::var_os("LLVM7_LD_LIBRARY_PATH")
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
    );
    if let Some(a) = std::env::var_os("LLVM7_LD_LIBRARY_PATH") {
        let new = if let Some(p) = std::env::var_os("LD_LIBRARY_PATH") {
            format!("{}:{}", p.to_str().unwrap(), a.to_str().unwrap())
        } else {
            String::from(a.to_str().unwrap())
        };
        std::env::set_var("LD_LIBRARY_PATH", &new);
    }

    //Make cargo re-run this if the common crate changed.
    println!("cargo:rerun-if-changed=../common");
    let mut builder = CudaBuilder::new("./kernel");

    //Specify nvidia build arch.
    builder.arch = NvvmArch::Compute50;

    // let out_dir = std::env::var_os("OUT_DIR").unwrap();
    // let dest_path = std::path::Path::new(&out_dir).join("kernel.ptx");
    let output = builder/*.copy_to(dest_path)*/.build().unwrap();
    println!(
        "cargo:rustc-env=KERNEL_PTX_PATH={}",
        output.as_path().display()
    );
}
