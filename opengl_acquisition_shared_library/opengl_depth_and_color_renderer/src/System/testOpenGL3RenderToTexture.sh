#!/bin/bash

rm gl3
gcc -o gl3Tex glx3.c glx_testRenderToTexture.c ../Rendering/ShaderPipeline/shader_loader.c ../Tools/save_to_file.c ../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGL -lGLU -lX11 -lGLEW
./gl3Tex

exit 0
