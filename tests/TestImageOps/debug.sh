#!/bin/bash
#--leak-check=yes --show-reachable=yes 
LD_LIBRARY_PATH=.:/home/ammar/Documents/Programming/input_acquisition/3dparty/libfreenect/build/lib valgrind --tool=exp-sgcheck --tool=memcheck  --num-callers=20 --track-fds=yes  ./TestImageOps 2> error.txt


exit 0
