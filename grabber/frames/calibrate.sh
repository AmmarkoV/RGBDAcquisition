#!/bin/bash

OPENCV_BIN="/home/ammar/Documents/3dParty/opencv-3.2.0/build/bin/"
ln -s $OPENCV_BIN/cpp-example-calibration
ln -s $OPENCV_BIN/cpp-example-imagelist_creator

./cpp-example-imagelist_creator toCalib.yaml $1
./cpp-example-calibration -w=9 -h=6 -s=0.021 toCalib.yaml

exit 0
