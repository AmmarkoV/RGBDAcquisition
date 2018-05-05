#!/bin/bash
 
#OPENNI2
#./run_viewer.sh -module OPENNI2 -from 0  -processor ../processors/Movidius/libMovidius.so  Movidius  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@

#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth  -processor ../processors/Movidius/libMovidius.so  Movidius  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@
  
./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth  -processor ../processors/Movidius/libMovidius.so  Movidius  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo-tiny.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/tiny-yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@
exit 0
