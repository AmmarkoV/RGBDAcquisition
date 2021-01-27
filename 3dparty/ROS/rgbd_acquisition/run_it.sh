#!/bin/bash

#roslaunch rgbd_acquisition rgbd_acquisition.launch deviceID:=/dev/video0 moduleID:=V4L2 width:=640 height:=480 framerate:=30


STARTDIR=`pwd`

#Switch to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


cd bin
./run_it.sh $@
cd ..

cd "$STARTDIR"

exit 0
