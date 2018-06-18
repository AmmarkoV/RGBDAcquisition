#!/bin/bash

rm gl3
gcc -o gl3 glx3.c glx_test.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGL -lX11 -lGLEW
./gl3

exit 0
