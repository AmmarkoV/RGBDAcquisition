#!/bin/bash

rm gl3MultiDiff
gcc -o gl3MultiDiff -DUSE_GLEW glx3.c glx_testMultiViewportDiff.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c ../ModelLoader/model_loader_tri.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../tools/AmMatrix/matrixOpenGL.c  ../../../../tools/Calibration/libCalibrationLibrary.a ../../../../acquisition/libRGBDAcquisition.so -lm -lGL -lGLU -lX11 -lGLEW
./gl3MultiDiff

exit 0
