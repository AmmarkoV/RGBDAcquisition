#!/bin/bash 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

DATASET=""

if (( $#<1 ))
then 
 echo "Please provide arguments first argument is dataset "
 exit 1
else
 DATASET=$1-data
fi
 
THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 

mkdir $DATASET

ffmpeg -i $1  -r 30 -q:v 1  $DATASET/colorFrame_0_%05d.jpg

cp $DATASET/colorFrame_0_00001.jpg $DATASET/colorFrame_0_00000.jpg

cd $DATASET
cd ..

cd $STARTDIR 
exit 0

