#!/bin/bash
STARTDIR=`pwd` 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  

CAFFE_DIR="/home/ammar/Documents/3dParty/Realtime_Multi-Person_Pose_Estimation/caffe_demo"

cd $CAFFE_DIR
echo "Will try to output results to $DIR/$1/dnnOut" 

echo "Will run : ./build/examples/rtpose/rtpose.bin --image_dir $DIR/$1/ --no_frame_drops --write_json $DIR/$1/dnnOut --logtostderr  "

./build/examples/rtpose/rtpose.bin --image_dir $DIR/$1/ --no_frame_drops --write_json $DIR/$1/dnnOut --logtostderr 
 
rm $DIR/$1/dnnOut/depth*
rm $DIR/$1/dnnOut/converted.csv

cd "$DIR"  

I=0
for s in $(ls $DIR/$1/dnnOut | sort | grep json); do  
  echo "Converting JSON files .. $s $I "
  ../../tools/Primitives/testSkeleton $DIR/$1/dnnOut/$s $I 2> /dev/null 1>>$DIR/$1/dnnOut/converted.csv
  ((I=I+1)) 
done  

exit 0
