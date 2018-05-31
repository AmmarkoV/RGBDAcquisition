#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

#essential stuff to build 
sudo apt-get install build-essential cmake

#jpg/png x11/OpenGL stuff for openGL plugin 
sudo apt-get install libx11-dev  freeglut3-dev libjpeg-dev libpng12-dev libglew-dev

#openCV for calibration and viewer
sudo apt-get install libcv-dev libopencv-dev

#Doxygen and documentation stuff
sudo apt-get install doxygen graphviz

cd ../3dparty
./get_third_party_libs.sh
cd ..

scripts/getBroadcastDependencies.sh

exit 0
