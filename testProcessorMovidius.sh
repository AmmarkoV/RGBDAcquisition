#!/bin/bash
 
#OPENNI2
#./run_viewer.sh -module OPENNI2 -from 0  -processor ../processors/Movidius/libMovidius.so  Movidius  --payload ./payload.sh $@ 

./run_viewer.sh -module V4L2 -from /dev/video0  -noDepth  -processor ../processors/Movidius/libMovidius.so  Movidius  --payload ./payload.sh $@
exit 0
