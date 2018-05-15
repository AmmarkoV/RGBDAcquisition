#!/bin/bash
  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

ln -s grabber/frames grabbed_frames 

cd grabber
ldd ../openni2_acquisition_shared_library/libOpenNiAquisition.so  | grep "not found"
ldd ../acquisition/libAcquisition.so  | grep "not found"
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes  ./Grabber $@ 2> ../error.txt
cd ..

exit 0
