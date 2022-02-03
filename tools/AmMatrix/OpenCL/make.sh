#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


sudo ln -s /usr/local/cuda/lib64/libOpenCL.so /usr/lib/libOpenCL.so

gcc mat_mul.c -o matmul -lOpenCL


exit 0
