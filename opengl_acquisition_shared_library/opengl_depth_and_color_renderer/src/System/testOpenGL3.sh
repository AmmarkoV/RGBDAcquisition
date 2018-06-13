#!/bin/bash

rm gl3
gcc -o gl3 glx3.c glx_test.c ../Rendering/ShaderPipeline/shader_loader.c  -lGL -lX11 -lGLEW
./gl3

exit 0
