#!/bin/bash

rm gl3Tex
gcc -o gl3Tex  -DUSE_GLEW glx3.c glx_testRenderToTexture.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/downloadFromRenderer.c ../Tools/save_to_file.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGL -lGLU -lX11 -lGLEW
./gl3Tex

exit 0
