#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd grabber
ldd ../openni2_acquisition_shared_library/libOpenNiAquisition.so  | grep "not found"
ldd ../acquisition/libAcquisition.so  | grep "not found"
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib ./Grabber $@
cd ..

exit 0
