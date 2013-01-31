#!/bin/bash
  
ln -s grabber/frames grabbed_frames 

cd grabber_mux
ldd ../openni2_acquisition_shared_library/libOpenNiAquisition.so  | grep "not found"
ldd ../acquisition/libAcquisition.so  | grep "not found"
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes  ./GrabberMux $@ 2>error.txt
cd ..

exit 0
