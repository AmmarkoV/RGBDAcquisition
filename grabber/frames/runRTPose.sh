#!/bin/bash
STARTDIR=`pwd` 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"  

CAFFE_DIR="/home/ammar/Documents/3dParty/Realtime_Multi-Person_Pose_Estimation/caffe_demo"

cd $CAFFE_DIR
echo "Will try to output results to $DIR/$1/dnnOut" 

./build/examples/rtpose/rtpose.bin --image_dir $DIR/$1/ --no_frame_drops --write_json $DIR/$1/dnnOut --logtostderr 
 
rm $DIR/$1/dnnOut/depth*

exit 0
