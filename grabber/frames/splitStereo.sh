#!/bin/bash



INPUTFILE="pnm"

if (( $#<1 ))
then 
 echo "Please provide arguments first argument is dataset ,  second is file format ( i.e. boxNew.mp4 ) "
 exit 1
else
 INPUTFILE=$1 
 DATASET=$1-data
fi
 

FULL_WIDTH=`ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 $INPUTFILE`
WIDTH=`expr $FULL_WIDTH / 2`
HEIGHT=`ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 $INPUTFILE`

echo "File size is $FULL_WIDTH x $HEIGHT"
echo "Each feed is $WIDTH"
#exit 0

mkdir $DATASET

ffmpeg -i $1  -r 30 -q:v 1 -filter:v "crop=$WIDTH:$HEIGHT:0:0"  $DATASET/colorFrame_0_%05d.jpg
 
cp $DATASET/colorFrame_0_00001.jpg $DATASET/colorFrame_0_00000.jpg


ffmpeg -i $1  -r 30 -q:v 1 -filter:v "crop=$WIDTH:$HEIGHT:$WIDTH:0"  $DATASET/colorFrame_1_%05d.jpg

cp $DATASET/colorFrame_1_00001.jpg $DATASET/colorFrame_1_00000.jpg
exit 0
