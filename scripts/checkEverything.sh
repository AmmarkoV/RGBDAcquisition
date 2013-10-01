#!/bin/bash


red=$(printf "\033[31m")
green=$(printf "\033[32m")
yellow=$(printf "\033[33m")
blue=$(printf "\033[34m")
magenta=$(printf "\033[35m")
cyan=$(printf "\033[36m")
white=$(printf "\033[37m")
normal=$(printf "\033[m")

normalChars=$(printf "\033[0m")
boldChars=$(printf "\033[1m")
underlinedChars=$(printf "\033[4m")
blinkingChars=$(printf "\033[5m") 


STARTDIR=`pwd`

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ..

toCheck="tools 3dparty scripts synergiesAdapter viewer libfreenect_acquisition_shared_library openni2_acquisition_shared_library openni1_acquisition_shared_library opengl_acquisition_shared_library opengl_acquisition_shared_library/opengl_depth_and_color_renderer openni1_acquisition_shared_library openni2_acquisition_shared_library template_acquisition_shared_library v4l2_acquisition_shared_library v4l2stereo_acquisition_shared_library viewer"

 for f in $toCheck
           do  
             if [ -d $f ]
              then
              echo "$green Found $f directory $normal" 
             else
              echo "$red Could not find $f directory $normal"
             fi
  done 


  echo "\n\n\n"

toCheck="tools/Undistort/undistort tools/ExtrinsicCalibration/extrinsicCalibration tools/DepthImagesConverter/DepthImagesConverter tools/Calibration/calibration tools/Calibration/libCalibrationLibrary.a acquisition/Acquisition.so  acquisition_mux/libAcquisitionMux.so acquisitionSegment/libacquisitionSegment.a"

 for f in $toCheck
           do  
             if [ -e $f ]
              then
              echo "$green $f built ok.. $normal" 
             else
              echo "$red $f failed .. $normal"
             fi
  done 

toCheck="grabber/Grabber viewer/Viewer grabber_segment/GrabberSegment grabber_mux/GrabberMux acquisitionBroadcast/acquisitionBroadcast synergiesAdapter/Adapter"

 for f in $toCheck
           do  
             if [ -e $f ]
              then
              echo "$green $f built ok.. $normal" 
             else
              echo "$red $f failed .. $normal"
             fi
  done 



cd "$STARTDIR"

exit 0
