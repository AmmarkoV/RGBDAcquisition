#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -e makehuman.tri ]
then 
 echo "Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/makehuman.tri
fi 

rm gl3MeshTransform
gcc -o gl3MeshTransform -DUSE_GLEW glx3.c glx_testStereoViewport.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c  ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c ../ModelLoader/model_loader_tri.c ../ModelLoader/model_loader_transform_joints.c ../TrajectoryParser/InputParser_C.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c ../../../../../tools/AmMatrix/simpleRenderer.c ../../../../../tools/AmMatrix/quaternions.c ../MotionCaptureLoader/bvh_loader.c ../MotionCaptureLoader/calculate/bvh_project.c ../MotionCaptureLoader/calculate/bvh_to_tri_pose.c ../MotionCaptureLoader/calculate/bvh_transform.c ../MotionCaptureLoader/edit/bvh_cut_paste.c ../MotionCaptureLoader/edit/bvh_filter.c ../MotionCaptureLoader/edit/bvh_interpolate.c ../MotionCaptureLoader/edit/bvh_merge.c ../MotionCaptureLoader/edit/bvh_randomize.c ../MotionCaptureLoader/edit/bvh_remapangles.c ../MotionCaptureLoader/edit/bvh_rename.c ../MotionCaptureLoader/export/bvh_export.c ../MotionCaptureLoader/export/bvh_to_bvh.c ../MotionCaptureLoader/export/bvh_to_c.c ../MotionCaptureLoader/export/bvh_to_csv.c ../MotionCaptureLoader/export/bvh_to_svg.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.c ../MotionCaptureLoader/ik/bvh_inverseKinematics.c ../MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.c ../MotionCaptureLoader/ik/levmar.c ../MotionCaptureLoader/import/fromBVH.c ../MotionCaptureLoader/metrics/bvh_measure.c  -lm -lGL -lGLU -lX11 -lGLEW -pthread
./gl3MeshTransform $@

exit 0