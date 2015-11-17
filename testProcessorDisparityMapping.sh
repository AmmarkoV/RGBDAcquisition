#!/bin/bash
 

./run_viewer.sh -module TEMPLATE -from firstGPSCar -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -maxFrames 2


exit 0
