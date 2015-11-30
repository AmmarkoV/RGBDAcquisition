#!/bin/bash
 
#  firstGPSCar
./run_viewer.sh -module TEMPLATE -from allimiagpscalibra -seek 60  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -maxFrames 200 -disparityCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/allimiagpscalibra/DisparityMappingCalibration  -disparitySADWindowSize 70 -disparitySwapColorFeeds


exit 0
