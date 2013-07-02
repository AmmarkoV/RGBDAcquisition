#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd acquisitionBroadcast
ldd acquisitionBroadcast  | grep not 
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./acquisitionBroadcast $@
cd ..

exit 0
