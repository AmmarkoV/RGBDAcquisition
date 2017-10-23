#!/bin/bash

killall rostopic
killall ros_dumper
sleep 1

rostopic echo -p /table > table&
rostopic echo -p  /armRange > armRange&
rostopic echo -p /clock > clock&
rostopic echo -p  /kinect > kinect &
rostopic echo -p  /shoulder > shoulder&
rostopic echo -p /shoulder_table > shoulder_table &
rostopic echo -p /table > table &
rostopic echo -p /twistAngle >twistAngle &
rostopic echo -p /wrist >wrist &
rostopic echo -p /wrist_table  >wrist_table &


roslaunch ros_dumper ros_dumper.launch &

echo " Almost Ready "
sleep 10
rosbag play --clock /home/ammar/Documents/Programming/RGBDAcquisition/grabber/frames/measurement1_short.bag -r 0.1

killall rostopic
killall ros_dumper

exit 0
