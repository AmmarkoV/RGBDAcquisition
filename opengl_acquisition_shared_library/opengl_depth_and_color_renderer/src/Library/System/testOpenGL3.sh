#!/bin/bash

rm gl3
gcc -o gl3  -DUSE_GLEW glx_test.c glx3.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGLEW -lGL -lX11 
./gl3

exit 0
