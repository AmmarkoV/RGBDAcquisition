#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd grabber
ldd Grabber  | grep not
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./Grabber $@
cd ..

exit 0
