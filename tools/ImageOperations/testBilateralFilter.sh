#!/bin/bash

STARTDIR=`pwd`  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$DIR" 

SRCIMG="../../media/samples/test.jpg"
SRCIMG="../../media/samples/lenna.png"
SRCIMG="../../media/samples/big.jpg"
WINDOWSIZE="7"

time ./imageopsutility $SRCIMG outputCTBilateral.jpg --ctbilateral 5.0 5

exit 0
time ./imageopsutility $SRCIMG outputDummy.jpg --sattest 

time ./imageopsutility $SRCIMG outputDer_s5.png --deriche 5.0 0
time ./imageopsutility $SRCIMG outputDer_s15.png --deriche 15.0 0
time ./imageopsutility $SRCIMG outputDer_Fs15.png --dericheF 15.0 0
time ./imageopsutility $SRCIMG outputDer_s25.png --deriche 25.0 0
timeout 5 gpicview outputDer_s5.png
exit 0

time ./imageopsutility $SRCIMG outputGray.jpg --monochrome  

time ./imageopsutility $SRCIMG outputMedian3x3.jpg --median 3 3
time ./imageopsutility $SRCIMG outputMedian8x8.jpg --median 8 8
#7minutes
time ./imageopsutility $SRCIMG outputMedian16x16.jpg --median 16 16
#This takes too long
#time ./imageopsutility $SRCIMG outputMedian32x32.jpg --median 32 32

time ./imageopsutility $SRCIMG outputMean3x3.jpg --meansat 3 3
time ./imageopsutility $SRCIMG outputMean8x8.jpg --meansat 8 8
time ./imageopsutility $SRCIMG outputMean16x16.jpg --meansat 16 16
time ./imageopsutility $SRCIMG outputMean32x32.jpg --meansat 32 32


exit 0

#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes  ./imageopsutilityD $SRCIMG test.jpg --sattest  2> error.txt 
#./imageopsutility $SRCIMG test.jpg --sattest 

time ./imageopsutility $SRCIMG output0.jpg --contrast 2.9 --bilateral 0.4 0.4 $WINDOWSIZE 
time ./imageopsutility $SRCIMG output1.jpg --contrast 2.9 --bilateral 1.0 1.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output2.jpg --contrast 2.9 --bilateral 5.0 6.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output3.jpg --contrast 2.9 --bilateral 25.0 26.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output4.jpg --contrast 2.9 --bilateral 125.0 126.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output5.jpg --contrast 2.9 --bilateral 195.0 196.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output6.jpg --contrast 2.9 --bilateral 225.0 226.0 $WINDOWSIZE
time ./imageopsutility $SRCIMG output7.jpg --contrast 2.9 --bilateral 255.0 256.0 $WINDOWSIZE

# - - - - - 
timeout 15 gpicview output0.jpg output1.jpg output2.jpg output3.jpg output4.jpg

exit 0
