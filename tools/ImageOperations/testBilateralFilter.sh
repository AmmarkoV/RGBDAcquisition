#!/bin/bash

STARTDIR=`pwd`  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$DIR" 

SRCIMG="../../media/samples/test.jpg"
WINDOWSIZE="5"
 
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes  ./imageopsutility $SRCIMG test.jpg --sattest  2> error.txt


./imageopsutility $SRCIMG test.jpg --sattest
exit 0

./imageopsutility $SRCIMG output0.jpg --contrast 2.9 --bilateral 0.4 0.4 $WINDOWSIZE
./imageopsutility $SRCIMG output1.jpg --contrast 2.9 --bilateral 1.0 1.0 $WINDOWSIZE
./imageopsutility $SRCIMG output2.jpg --contrast 2.9 --bilateral 5.0 6.0 $WINDOWSIZE
./imageopsutility $SRCIMG output3.jpg --contrast 2.9 --bilateral 25.0 26.0 $WINDOWSIZE
./imageopsutility $SRCIMG output4.jpg --contrast 2.9 --bilateral 125.0 126.0 $WINDOWSIZE
./imageopsutility $SRCIMG output5.jpg --contrast 2.9 --bilateral 195.0 196.0 $WINDOWSIZE

# - - - - - 
timeout 15 gpicview output0.jpg output1.jpg output2.jpg output3.jpg output4.jpg

exit 0
