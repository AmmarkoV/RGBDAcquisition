#!/bin/bash
 
 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" 

cd redist
ln -s ../3dparty/darknet/data
cd ..

cd .. 
ln -s $DIR/3dparty
cd $DIR



#YOLOv3
#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolov3.weights $DIR/3dparty/darknet/cfg/yolov3.cfg  $DIR/3dparty/darknet/cfg/coco.data $DIR/3dparty/darknet/data/coco.names --payload ./payload.sh $@


#YOLO
#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolo.weights $DIR/3dparty/darknet/cfg/yolo.cfg  $DIR/3dparty/darknet/cfg/coco.data $DIR/3dparty/darknet/data/coco.names --payload ./payload.sh $@
 
#CIFAR
#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/backup/cifar_small.weights $DIR/3dparty/darknet/cfg/cifar_small.cfg  $DIR/3dparty/darknet/cfg/cifar.data  $DIR/3dparty/darknet/data/cifar/labels.txt --payload ./payload.sh $@

#TINYYOLO
#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/yolov2-tiny.weights $DIR/3dparty/darknet/cfg/yolov2-tiny.cfg $DIR/3dparty/darknet/data/coco.names --noVisualization --noFileOutput --payload ./payload.sh $@


#IMAGENET 
./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  $DIR/3dparty/darknet/darknet19.weights $DIR/3dparty/darknet/cfg/darknet19.cfg  $DIR/3dparty/darknet/cfg/imagenet1k.data  $DIR/3dparty/darknet/data/imagenet.names --classifier --payload ./payload.sh $@

#Run in editor
#./run_editor.sh /home/ammar/Documents/Programming/RGBDAcquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/RGBDAcquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/RGBDAcquisition/3dparty/darknet/cfg/coco.data /home/ammar/Documents/Programming/RGBDAcquisition/3dparty/darknet/data/coco.names


exit 0 
