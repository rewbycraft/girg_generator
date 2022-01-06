#!/bin/bash
cd "$(dirname "$0")"
set -exuo pipefail
mkdir -p empty
docker build -t cuda-builder:latest --network host -f builder.Dockerfile empty
docker run -it --rm --network host -v "$(pwd):/project" --workdir /project cuda-builder:latest /bin/bash

