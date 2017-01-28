#!/bin/bash


FROM="$1" 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd $FROM


../../../tools/ExtrinsicCalibration/extrinsicCalibration -v -w 9 -h 13 -s 0.07 -c color.calib -i colorFrame_0_00000.pnm

cp color.calib depth.calib

cd ..

cd $STARTDIR 

exit 0
