#!/bin/bash
 
#firstGPSCar
./run_viewer.sh -module TEMPLATE -from allimiagpscalibra  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -maxFrames 200


exit 0
