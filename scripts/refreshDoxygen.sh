#!/bin/bash

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
 
export RGBD_DOXYGEN_INPUT="$DIR/.."
export RGBD_DOXYGEN_OUTPUT="$DIR/../doc"

cd ..
doxygen doc/doxyfile
cd doc/latex
make
cd .. 
cd ..

ln -s doc/latex/refman.pdf RGBDAcquisition.pdf


cd "$STARTDIR"
 

exit 0
