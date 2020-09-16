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
FRAMERATER=`ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=s=x:p=0 $INPUTFILE`
FRAMERATE=`expr $FRAMERATER`

echo "File size is $FULL_WIDTH x $HEIGHT"
echo "Each feed is $WIDTH"
#exit 0

mkdir -p $DATASET/calib

echo "Left feed is 0_color and Right 1_color" > $DATASET/README

#---------------------------------------------------------------------
mkdir -p $DATASET/0_color
ffmpeg -i $1  -r $FRAMERATE -q:v 1 -filter:v "crop=$WIDTH:$HEIGHT:0:0"  $DATASET/0_color/%05d.jpg
cp $DATASET/0_color/00001.jpg $DATASET/0_color/00000.jpg


#---------------------------------------------------------------------
mkdir -p $DATASET/1_color
ffmpeg -i $1  -r $FRAMERATE -q:v 1 -filter:v "crop=$WIDTH:$HEIGHT:$WIDTH:0"  $DATASET/1_color/%05d.jpg
cp $DATASET/1_color/00001.jpg $DATASET/1_color/00000.jpg


#---------------------------------------------------------------------
exit 0
