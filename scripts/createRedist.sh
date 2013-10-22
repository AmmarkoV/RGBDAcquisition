#!/bin/bash

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..


mkdir redist


cd redist
ln -s 

binaries="grabber/Grabber viewer/Viewer grabber_segment/GrabberSegment grabber_mux/GrabberMux acquisitionBroadcast/acquisitionBroadcast synergiesAdapter/Adapter"

 for f in $binaries
           do  
             if [ -e "../$f" ]
              then
                ln -s "../$f"
             else
              echo "$red "../$f" does not exist .. $normal"
             fi
  done 


libraries="acquisition/libAcquisition.so libfreenect_acquisition_shared_library/libFreenectAcquisition.so opengl_acquisition_shared_library/libOpenGLAcquisition.so openni1_acquisition_shared_library/libOpenNI1Acquisition.so openni2_acquisition_shared_library/libOpenNI2Acquisition.so template_acquisition_shared_library/libTemplateAcquisition.so v4l2_acquisition_shared_library/libV4L2Acquisition.so v4l2stereo_acquisition_shared_library/libV4L2StereoAcquisition.so"

 for f in $libraries
           do  
             if [ -e "../$f" ]
              then
                ln -s "../$f"
             else
              echo "$red "../$f" does not exist .. $normal"
             fi
  done 



ln -s ../grabber/frames
ln -s ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Models
ln -s ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Scenes
ln -s ../acquisitionBroadcast/AmmarServer/public_html
 
../3dparty/link_to_libs.sh ../3dparty/

cd ..
#At root dir
ln -s grabber/frames grabbed_frames

cd opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Renderer/
ln -s ../../libOpenGLAcquisition.so


cd "$STARTDIR"
 
exit 0
