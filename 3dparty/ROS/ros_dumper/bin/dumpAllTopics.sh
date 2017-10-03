#!/bin/bash


rostopic echo /kinect > kinect.dump&
rostopic echo /shoulder > shoulder.dump&
rostopic echo /shoulder_table > shoulder_table.dump&
rostopic echo /wrist > wrist.dump&
rostopic echo /wrist_table > wrist_table.dump&
rostopic echo /clock > clock.dump&
rostopic echo /twistAngle > twistAngle.dump&



exit 0
