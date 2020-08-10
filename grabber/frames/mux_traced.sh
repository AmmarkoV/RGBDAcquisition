#!/bin/bash 
#./mux_traced.sh blue.webm-data/raytraced %04d.png muxHD-blue.webm-data-19-04-20_04-49-39.mp4


STARTDIR=`pwd`
#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
 
if (( $#<3 ))
then 
 echo "Please provide arguments first argument is dataset ,  second is file format ( i.e. boxNew jpg ) "
 exit 1
else
 DATASET_A=$1
 PATTERN_A=$2
 FILE_B=$3 
fi

echo "Dataset A is $DATASET_A/$PATTERN_A "
echo "Dataset B is $FILE_B "
 
THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"` 

#FILTER=" -filter_complex '[1]split[m][a]; [a]geq='if(gt(lum(X,Y),16),255,0)',hue=s=0[al]; [m][al]alphamerge[ovr]; [0][ovr]overlay' " 
#FILTER=' -filter_complex "overlay" ' 

# -hwaccel -crf 5

#ffmpeg -framerate 30 -i $DATASET_A/$PATTERN_A -framerate 30 -i $DATASET_B/$PATTERN_B  -filter_complex "[1]split[m][a]; [a]geq='if(gt(lum(X,Y),32),255,0)',hue=s=0[al]; [m][al]alphamerge[ovr]; [0][ovr]overlay" -strict -2 -y -r 30 -threads 8 -preset slow -f webm -vcodec libvpx-vp9  -vb 2048k -pix_fmt yuv420p  ./muxHD-$DATASET_A-$THEDATETAG.webm
 
#ffmpeg -framerate 30 -i $DATASET_A/$PATTERN_A  -i $FILE_B  -filter_complex "[1]split[m][a]; [a]geq='if(gt(lum(X,Y),32),255,0)',hue=s=0[al]; [m][al]alphamerge[ovr]; [0][ovr]overlay" -y -r 30 -threads 8  -pix_fmt yuv420p -crf 18 ./traHD-$THEDATETAG-$FILE_B.mp4
 
ffmpeg  -i $FILE_B  -framerate 30 -i $DATASET_A/$PATTERN_A -filter_complex "[0:v]scale=1920:-1[bg];[bg][1:v]overlay=(main_w-overlay_w):(main_h-overlay_h)" -y -r 30 -threads 8  -pix_fmt yuv420p -crf 18 ./traHD-$THEDATETAG-$FILE_B.mp4
 

#./mux_videos.sh GOPR3229.MP4-data visualization/colorFrame_0_%05d_rendered.png bvhRendering_GOPR3229.MP4-data colorFrame_0_%05d.jpg

cd $STARTDIR 
exit 0

