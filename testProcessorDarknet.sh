#!/bin/bash
 
 
./run_viewer.sh -module TEMPLATE -from naobasic  -noDepth -noColor   -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  $@
 
#./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth   -processor ../processors/DarknetProcessor/libDarknetProcessor.so  DarknetProcessor  /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/yolo.weights /home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/darknet/cfg/yolo.cfg  $@s

exit 0
