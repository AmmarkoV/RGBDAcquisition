#!/bin/bash

#essential stuff to build 
sudo apt-get install build-essential cmake

#jpg/png x11/OpenGL stuff for openGL plugin 
sudo apt-get install libx11-dev  freeglut3-dev libjpeg-dev libpng12-dev

#openCV for calibration and viewer
sudo apt-get install libcv-dev libopencv-dev

cd 3dparty
./get_third_party_libs.sh
cd ..

scripts/getBroadcastDependencies.sh

exit 0
