#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd grabber_segment 
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./GrabberSegment $@
cd ..

exit 0
