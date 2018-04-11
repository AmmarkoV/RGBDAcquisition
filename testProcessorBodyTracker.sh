#!/bin/bash
  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd redist 

LIBS="../processors/BodyTracker/forth_skeleton_tracker_redist"

#LD_PRELOAD=$LIBS/libboost_system.so.1.57.0:$LIBS/libAcquisition.so:$LIBS/libCore.so:$LIBS/../libBodyTracker.so 
LD_LIBRARY_PATH=/home/ammar/Documents/Programming/RGBDAcquisition/processors/BodyTracker/forth_skeleton_tracker_redist/:/home/ammar/Downloads/opencvlib:.  ./Viewer -module OPENNI2 -from 0 -processor /home/ammar/Documents/Programming/RGBDAcquisition/processors/BodyTracker/libBodyTracker.so  BodyTracker $@
  
exit 0
