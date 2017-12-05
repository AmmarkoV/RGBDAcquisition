#!/bin/bash
 
  
./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth  -processor ../processors/Movidius/libMovidius.so  Movidius  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@
  
exit 0
