#!/bin/bash
 
 
#./run_viewer.sh -module TEMPLATE -from naobasic  -noDepth  -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data $@
 
./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data --payload ./payload.sh $@
 
#./run_viewer.sh -module DESKTOP -from /dev/video0  -noDepth -noColor  -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/coco.data $@
exit 0
