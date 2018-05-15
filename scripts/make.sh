#!/bin/bash
 
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

mkdir build
cd build
cmake ..

make
notify-send "Done Compiling RGBDAcquisition"


cd "$DIR"
cd ..
mkdir -p opengl_acquisition_shared_library/opengl_depth_and_color_renderer/build
cd opengl_acquisition_shared_library/opengl_depth_and_color_renderer/build

make 
notify-send "Done Compiling OpenGLRenderer"

cd "$DIR"

exit 0
