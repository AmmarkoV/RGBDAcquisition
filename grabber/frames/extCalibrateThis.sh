#!/bin/bash


FROM="$1" 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd $FROM


../../../tools/ExtrinsicCalibration/extrinsicCalibration -v -w 6 -h 9 -s 0.02 -c color.calib -i colorFrame_0_00000.pnm

cp color.calib depth.calib

cd ..

cd $STARTDIR 

exit 0
