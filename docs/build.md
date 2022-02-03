# Build Setup

## Docker build environment (for CI, just producing a binary)

Make sure you have `docker` installed and your local user is able to execute docker commands without sudo.

Then run:
```shell
./builder.sh
```

This will give you a shell that is configured right to build this software.
See "Build the software" below for how to build binaries.

## Local build environment (for development, testing and benchmarks)

Documentation based on Ubuntu 20.04.
Should be valid for derivatives thereof or easily adapted to other distros.

### Installing cuda
There are other ways to install cuda, it is important that it includes libnvvm3.
You will have to adjust the path to libnvvm3 used later in this file if you use another method.

```shell
# Install programs to add repos
sudo apt-get update
sudo apt-get install -y \
    curl \
    gnupg \
    software-properties-common

# Add the nvidia repo
sudo curl -o /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && rm -rf /var/lib/apt/lists/*

# Install packages
sudo apt-get update
sudo apt-get install -y \
    cuda-11-3 \
    libnvvm3 \
    tmux
```

### Installing other dependencies

Note that llvm needs to be version 7!
The guide assumes your build is dynamically linked (true for the ubuntu packages).

```shell
# Install packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    llvm-7
```

### Install rust
As long as you have rust installed via `rustup`, it will automatically download the right toolchain and components.
```shell
curl -o /tmp/rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    && chmod +x /tmp/rustup.sh \
    && /tmp/rustup.sh -y -c "rust-src" -c "rustc-dev" -c "llvm-tools-preview" \
    && rm /tmp/rustup.sh
```

### Setup environment variables
Add the following to your `$HOME/.profile`:
```shell
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/nvvm/lib64:$HOME/.rustup/toolchains/nightly-2021-12-04-x86_64-unknown-linux-gnu/lib/"
export LLVM_CONFIG=/usr/bin/llvm-config-7
export LLVM_LINK_SHARED=1
```

### Build the software
You should now be able to build the software.

#### Produce a binary
```shell
cargo build --release
```
The program binary will be in `target/release/`.

#### Run tests
```shell
cargo test
```

#### Run benchmarks
```shell
cargo bench --features benchmark
```
The results and reports will be in `target/criterion/`.

#### Setup an IDE
This software has been developed using Intellij IDEA with the Rust plugin.
