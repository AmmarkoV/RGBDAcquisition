#!/bin/bash

cd ~/catkin_ws/src/camera_broadcast/src/
gcc test.c protocol.c tools.c network.c AmmClient.c -o test



exit 0
