#!/bin/bash
 
 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

cd redist
ln -s ../3dparty/darknet/data
cd ..

cd .. 
ln -s $DIR/3dparty
cd $DIR


./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolo.weights $DIR/3dparty/darknet/cfg/yolo.cfg  $DIR/3dparty/darknet/cfg/coco.data $DIR/3dparty/darknet/data/coco.names --payload ./payload.sh $@

#./run_viewer.sh -module TEMPLATE -from naobasic  -noDepth  -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolo.weights $DIR/3dparty/darknet/cfg/yolo.cfg  $DIR/3dparty/darknet/cfg/coco.data $@
 


#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/backup/cifar_small.weights $DIR/3dparty/darknet/cfg/cifar_small.cfg  $DIR/3dparty/darknet/cfg/cifar.data --payload ./payload.sh $@

exit 0 
