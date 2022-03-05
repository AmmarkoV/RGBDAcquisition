#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -e makehuman.tri ]
then 
 echo "Body Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/makehuman.tri
fi 

if [ -e hair.tri ]
then 
 echo "Hair Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/hair.tri
fi 

if [ -e teeth.tri ]
then 
 echo "Teeth Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/teeth.tri
fi 

if [ -e tongue.tri ]
then 
 echo "Tongue Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/tongue.tri
fi 

if [ -e eyes.tri ]
then 
 echo "Eye Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/eyes.tri
fi 

if [ -e eyebrows.tri ]
then 
 echo "Eyebrows Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/eyebrows.tri
fi 

if [ -e eyelashes.tri ]
then 
 echo "Eyelashes Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/eyelashes.tri
fi 

if [ -e axis.tri ]
then 
 echo "Axis Model exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/axis.tri
fi 

if [ -e 01_02.bvh ]
then 
 echo "BVH Sample motion exists.."
else 
 cd "$DIR"  
 wget http://ammar.gr/mocapnet/mnet4/01_02.bvh
fi 


rm gl3MeshTransform
gcc -o gl3MeshTransform -DUSE_GLEW glx3.c glx_testMeshTransform.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c ../ModelLoader/model_loader_tri.c ../ModelLoader/model_loader_transform_joints.c ../TrajectoryParser/InputParser_C.c ../../../../../tools/Codecs/ppmInput.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c ../../../../../tools/AmMatrix/simpleRenderer.c ../../../../../tools/AmMatrix/quaternions.c ../MotionCaptureLoader/bvh_loader.c ../MotionCaptureLoader/calculate/bvh_project.c ../MotionCaptureLoader/calculate/bvh_to_tri_pose.c ../MotionCaptureLoader/calculate/bvh_transform.c ../MotionCaptureLoader/edit/bvh_cut_paste.c ../MotionCaptureLoader/edit/bvh_filter.c ../MotionCaptureLoader/edit/bvh_interpolate.c ../MotionCaptureLoader/edit/bvh_merge.c ../MotionCaptureLoader/edit/bvh_randomize.c ../MotionCaptureLoader/edit/bvh_remapangles.c ../MotionCaptureLoader/edit/bvh_rename.c ../MotionCaptureLoader/export/bvh_export.c ../MotionCaptureLoader/export/bvh_to_bvh.c ../MotionCaptureLoader/export/bvh_to_c.c ../MotionCaptureLoader/export/bvh_to_json.c ../MotionCaptureLoader/export/bvh_to_csv.c ../MotionCaptureLoader/export/bvh_to_svg.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.c ../MotionCaptureLoader/ik/bvh_inverseKinematics.c ../MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.c ../MotionCaptureLoader/ik/levmar.c ../MotionCaptureLoader/import/fromBVH.c ../MotionCaptureLoader/metrics/bvh_measure.c  -lm -lGL -lGLU -lX11 -lGLEW -pthread

./gl3MeshTransform $@


#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./gl3MeshTransform $@ 2>error.txt

exit 0
