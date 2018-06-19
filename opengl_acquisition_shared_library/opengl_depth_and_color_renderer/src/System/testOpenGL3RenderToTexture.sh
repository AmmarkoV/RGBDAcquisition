#!/bin/bash

rm gl3
gcc -o gl3Tex glx3.c glx_testRenderToTexture.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGL -lX11 -lGLEW
./gl3Tex

exit 0
