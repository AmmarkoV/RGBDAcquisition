#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd grabber_mux
ldd GrabberMux  | grep not 
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./GrabberMux $@
cd ..

exit 0
