#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


if [ -e ../../../Models/Ammar.tri ]
then 
 echo "Model exists.."
else
 cd ../../../Models/
 wget http://ammar.gr/models/Ammar.tri 
 cd "$DIR"
fi 

#-g
rm gl3MultiDiff
gcc  -o gl3MultiDiff -DUSE_GLEW glx3.c glx_testMultiViewportDiff.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c  ../ModelLoader/model_loader_tri.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/quaternions.h  ../../../../../tools/AmMatrix/matrixOpenGL.c  ../../../../../tools/Calibration/libCalibrationLibrary.a ../../../../../acquisition/libRGBDAcquisition.so -lm -lGL -lGLU -lX11 -lGLEW


ln -s ../../../../../acquisition/libRGBDAcquisition.so
ln -s ../../../../../template_acquisition_shared_library/libTemplateAcquisition.so
ln -s ../../../../../grabber/frames

LD_LIBRARY_PATH=. ./gl3MultiDiff  -from humanTS > /dev/null


exit 0
