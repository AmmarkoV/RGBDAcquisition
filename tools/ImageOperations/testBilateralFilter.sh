#!/bin/bash

STARTDIR=`pwd`  
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$DIR" 

echo "New Running Batch" > results.txt

rm *.jpg

WINDOWSIZE="7"

SRCPATH="../../media/samples/" 
SRCIMG="test.jpg"
SRCIMG="lenna.png"

SRCIMG="lenna.png"
time ./imageopsutility $SRCPATH/$SRCIMG median3x3-$SRCIMG --median 3 3     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median8x8-$SRCIMG --median 8 8     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median16x16-$SRCIMG --median 16 16 >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s5.png-$SRCIMG --deriche 5.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s15.png-$SRCIMG --deriche 15.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s25.png-$SRCIMG --deriche 25.0 0    >> results.txt


SIGMA="5.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt
SIGMA="15.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt


SIGMA="5.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="15"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 
 
SRCIMG="elina.jpg"
time ./imageopsutility $SRCPATH/$SRCIMG median3x3-$SRCIMG --median 3 3     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median8x8-$SRCIMG --median 8 8     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median16x16-$SRCIMG --median 16 16 >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s5.png-$SRCIMG --deriche 5.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s15.png-$SRCIMG --deriche 15.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s25.png-$SRCIMG --deriche 25.0 0    >> results.txt

SIGMA="5.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt
SIGMA="15.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt


SIGMA="5.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="15"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 

 


SRCIMG="big.jpg"
time ./imageopsutility $SRCPATH/$SRCIMG median3x3-$SRCIMG --median 3 3     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median8x8-$SRCIMG --median 8 8     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG median16x16-$SRCIMG --median 16 16 >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s5.png-$SRCIMG --deriche 5.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s15.png-$SRCIMG --deriche 15.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG deriche_s25.png-$SRCIMG --deriche 25.0 0    >> results.txt

SIGMA="5.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt
SIGMA="15.0"
time ./imageopsutility $SRCPATH/$SRCIMG fullBilateral-sigma=$SIGMA-$SRCIMG --bilateral $SIGMA $SIGMA $WINDOWSIZE    >> results.txt


SIGMA="5.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="5"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


SIGMA="15.0"
BINS="15"
time ./imageopsutility $SRCPATH/$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 1  >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG nulBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --ctbilateral $SIGMA $BINS 2    >> results.txt 
 ./imageopsutility derBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG norBilateral-sigma=$SIGMA-bins=$BINS-$SRCIMG --compare >> results.txt 


exit 0

echo "Now Running Intermediate Filters" > results.txt


time ./imageopsutility $SRCPATH/$SRCIMG outputDummy.jpg --sattest     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputGauss.jpg --gaussian    >> results.txt
 
time ./imageopsutility $SRCPATH/$SRCIMG outputDer_s5.png --deriche 5.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputDer_Fs15.png --dericheF 15.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputDer_s15.png --deriche 15.0 0    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputDer_s25.png --deriche 25.0 0    >> results.txt
timeout 5 gpicview outputDer_s5.png


time ./imageopsutility $SRCPATH/$SRCIMG outputGray.jpg --monochrome      >> results.txt

#This takes too long
#time ./imageopsutility $SRCPATH/$SRCIMG outputMedian32x32.jpg --median 32 32  

time ./imageopsutility $SRCPATH/$SRCIMG outputMean3x3.jpg --meansat 3 3    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputMean8x8.jpg --meansat 8 8    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputMean16x16.jpg --meansat 16 16    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG outputMean32x32.jpg --meansat 32 32    >> results.txt
 
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes  ./imageopsutilityD $SRCIMG test.jpg --sattest  2> error.txt 
#./imageopsutility $SRCIMG test.jpg --sattest 

time ./imageopsutility $SRCPATH/$SRCIMG output0.jpg --contrast 2.9 --bilateral 0.4 0.4 $WINDOWSIZE     >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output1.jpg --contrast 2.9 --bilateral 1.0 1.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output2.jpg --contrast 2.9 --bilateral 5.0 6.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output3.jpg --contrast 2.9 --bilateral 25.0 26.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output4.jpg --contrast 2.9 --bilateral 125.0 126.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output5.jpg --contrast 2.9 --bilateral 195.0 196.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output6.jpg --contrast 2.9 --bilateral 225.0 226.0 $WINDOWSIZE    >> results.txt
time ./imageopsutility $SRCPATH/$SRCIMG output7.jpg --contrast 2.9 --bilateral 255.0 256.0 $WINDOWSIZE    >> results.txt

exit 0

# - - - - - 
timeout 15 gpicview output0.jpg output1.jpg output2.jpg output3.jpg output4.jpg

exit 0
