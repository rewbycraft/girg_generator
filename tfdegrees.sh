#!/bin/bash
set -exuo pipefail
cat degrees.csv| cut -d, -f2 | tail -n +2 > degrees.txt

