#!/bin/bash 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

DATASET=""
EXTENSION="pnm"

if (( $#<1 ))
then 
 echo "Will run as if you supplied 7 as an argument"
else
 DATASET=$1
 EXTENSION=$2
fi
 
THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 

cd $DATASET
avconv -i colorFrame_0_%05d.$EXTENSION -y -r 20 -threads 8 -b 30000k -s 640x480  ../outHD_$THEDATETAG.mp4 
cd ..

cd $STARTDIR 
exit 0

