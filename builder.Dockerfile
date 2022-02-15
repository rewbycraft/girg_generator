FROM ubuntu:20.04

# Install package install tools.
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the nvidia repo
RUN curl -o /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && rm -rf /var/lib/apt/lists/*

# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cuda-11-3 \
    libssl-dev \
    llvm-7 \
    llvm-7-dev \
    llvm-7-tools \
    llvm-7-runtime \
    libnvvm3 \
    && rm -rf /var/lib/apt/lists/*

# Setup the llvm alias.
RUN ln -s /usr/bin/llvm-config-7 /usr/local/bin/llvm-config

# Install rustup.
RUN curl -o /tmp/rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    && chmod +x /tmp/rustup.sh \
    && /tmp/rustup.sh --default-toolchain nightly-2021-12-04-x86_64-unknown-linux-gnu -y -v -c "rust-src" -c "rustc-dev" -c "llvm-tools-preview" \
    && rm /tmp/rustup.sh

# Add libraries to the library search path.
RUN echo /usr/local/cuda-11.3/nvvm/lib64 > /etc/ld.so.conf.d/900_nvvm.conf && ldconfig
RUN echo /root/.rustup/toolchains/nightly-2021-12-04-x86_64-unknown-linux-gnu/lib/ > /etc/ld.so.conf.d/800_rust.conf && ldconfig

ENV LLVM_LINK_SHARED=1
