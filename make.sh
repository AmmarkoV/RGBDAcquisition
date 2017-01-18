#!/bin/bash
 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

mkdir build
cd build

cmake ..

make
notify-send "Done Compiling RGBDAcquisition"


cd /home/ammar/Documents/Programming/FORTH/input_acquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/build
make 
notify-send "Done Compiling OpenGLRenderer"

cd ..

exit 0
