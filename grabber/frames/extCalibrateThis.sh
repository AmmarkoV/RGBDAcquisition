#!/bin/bash


FROM="$1" 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd $FROM
 


FILES_TO_CONVERT=`ls | grep color`
for f in $FILES_TO_CONVERT
do 
../../../tools/ExtrinsicCalibration/extrinsicCalibration  -w 7 -h 4 -s 0.03 -c color.calib -i $f
#-v

done


cp color.calib depth.calib

cd ..

cd $STARTDIR 

exit 0
