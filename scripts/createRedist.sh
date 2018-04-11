#!/bin/bash

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..
 
mkdir redist
 
cd redist
ln -s 

ln -s ../editor/default.bmp 


ln -s ../../3dparty/librealsense/build/devel/lib/librealsense.so 

binaries="grabber/Grabber viewer/Viewer grabber_segment/GrabberSegment grabber_mux/GrabberMux acquisitionBroadcast/acquisitionBroadcast synergiesAdapter/Adapter editor/Editor"

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



ln -s ../editor/default.bmp default.bmp

ln -s ../grabber/frames
ln -s ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Models
ln -s ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Scenes
ln -s ../acquisitionBroadcast/AmmarServer/public_html
 
../3dparty/link_to_libs.sh ../3dparty/

cd ..

#At root dir
  
cd "$STARTDIR"
cd scripts
BINARIES_THAT_NEED_ACQUISITIONLIB="viewer grabber grabber_segment editor acquisitionBroadcast"
pwd

for f in $BINARIES_THAT_NEED_ACQUISITIONLIB
           do  
             if [ -d ../$f/ ]
              then
               cd ../$f/ 
                    
               ln -s ../acquisition/libAcquisition.so

               cd ../scripts/
             else
              echo "Could not create links for ../$f/ "
             fi
           done
cd ..



#At root dir
ln -s grabber/frames grabbed_frames

cd opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Renderer/
ln -s ../../libOGLRendererSandbox.so



cd "$STARTDIR"
 
exit 0
