#!/bin/bash
 
roslaunch openni_launch openni.launch &

sleep 10

rosrun rqt_reconfigure rqt_reconfigure 

exit 0
