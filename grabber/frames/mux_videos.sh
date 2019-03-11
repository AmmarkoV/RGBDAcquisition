#!/bin/bash 

STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
 
if (( $#<4 ))
then 
 echo "Please provide arguments first argument is dataset ,  second is file format ( i.e. boxNew jpg ) "
 exit 1
else
 DATASET_A=$1
 PATTERN_A=$2
 DATASET_B=$3
 PATTERN_B=$4
fi

echo "Dataset A is $DATASET_A/$PATTERN_A "
echo "Dataset B is $DATASET_B/$PATTERN_B "
 
THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 

#FILTER=" -filter_complex '[1]split[m][a]; [a]geq='if(gt(lum(X,Y),16),255,0)',hue=s=0[al]; [m][al]alphamerge[ovr]; [0][ovr]overlay' " 
#FILTER=' -filter_complex "overlay" ' 


ffmpeg -framerate 30 -i $DATASET_A/$PATTERN_A -framerate 30 -i $DATASET_B/$PATTERN_B  -filter_complex "[1]split[m][a]; [a]geq='if(gt(lum(X,Y),24),255,0)',hue=s=0[al]; [m][al]alphamerge[ovr]; [0][ovr]overlay" -strict -2 -y -r 30 -threads 8 -crf 10 -pix_fmt yuv420p ./muxHD-$DATASET-$THEDATETAG.webm
 
#./mux_videos.sh GOPR3229.MP4-data visualization/colorFrame_0_%05d_rendered.png bvhRendering_GOPR3229.MP4-data colorFrame_0_%05d.jpg

cd $STARTDIR 
exit 0

