#!/bin/bash

DIRECTORY="/home/ammar/Documents/3dParty/ncappzoo/caffe/TinyYolo"

mvNCCompile $DIRECTORY/tiny-yolo-v1.prototxt -w $DIRECTORY/tiny-yolo-v1_53000.caffemodel -s 12

exit 0
