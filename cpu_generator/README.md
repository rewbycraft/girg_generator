# CPU generator implementation

## Installing rust
See instructions on https://rustup.rs/

## Running the node_degrees tool
Replace `--help` with the arguments you wish to use after reading `--help`.
```sh
cargo run --package cpu_generator --release --bin node_degrees -- --help
```

### Defaults

Default values:
- Number of parallel workers: 4
- Size of block: 5000
- Number of vertices: 100000
- Dimensions: 2
- alpha (of the girg probability): 1.5
- beta (alpha of the weights pareto distribution): 1.5
- x_min (of the weights pareto distribution): 1.0
- Files get written to current directory

If the defaults are fine, run it as follows:
```sh
cargo run --package cpu_generator --release --bin node_degrees
```
