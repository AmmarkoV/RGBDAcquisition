#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

cd ~/catkin_ws/src/camera_broadcast/src/
gcc test.c protocol.c tools.c network.c AmmClient.c -o test



exit 0
