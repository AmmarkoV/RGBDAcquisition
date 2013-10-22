#!/bin/bash

FROM="fuse09"
TO="fuse09Segmented"
FRAMESNUM="534"
#FRAMESNUM="10"

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

THRESHOLD="5"

../../../run_segmenter.sh -from $FROM -to $TO -maxFrames $FRAMESNUM -eraseRGB 255 255 255 -minDepth 400 -maxDepth 800 -bbox -2000 -1650 -115 600 750 294 -calibration frames/$FROM/color.calib -combine depth
 
#-floodEraseDepthSource 529 290 $THRESHOLD -floodEraseDepthSource 529 330 $THRESHOLD -floodEraseDepthSource 529 350 $THRESHOLD -floodEraseDepthSource 529 390 $THRESHOLD -floodEraseDepthSource 529 430 $THRESHOLD 

cp color.calib ../$TO/color.calib
cp depth.calib ../$TO/depth.calib

cd $STARTDIR 

exit 0
