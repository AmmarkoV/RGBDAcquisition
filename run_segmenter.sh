#!/bin/bash
  

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


ln -s grabber/frames grabbed_frames 

cd grabber_segment 
ldd GrabberSegment | grep not
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./GrabberSegment $@
cd ..


cd $STARTDIR 

exit 0
