#!/bin/bash

rm shadertoy
gcc -o shadertoy ../System/glx3.c shadertoy.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/downloadFromRenderer.c ../Tools/save_to_file.c ../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../tools/AmMatrix/matrixOpenGL.c -lm -lGL -lGLU -lX11 -lGLEW
./shadertoy

exit 0
