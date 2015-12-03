#!/bin/bash
 
#  firstGPSCar firstGPSCar2 allimiagpscalibra
./run_viewer.sh -module TEMPLATE -from firstGPSCar2 -seek 60  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -maxFrames 200 -disparityCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/allimiagpscalibra/DisparityMappingCalibration  -disparitySADWindowSize 70 -disparitySwapColorFeeds $@



#./run_viewer.sh -module TEMPLATE -from oldStereoSet -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -maxFrames 200  -disparitySADWindowSize 25 $@


WIDTH="752"
HEIGHT="416"
FPS="30"

#./run_viewer.sh -module V4L2STEREO -from /dev/video0,/dev/video1 -resolution $WIDTH $HEIGHT   -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -disparitySpeckleRange 128  -disparitySADWindowSize 25 $@


exit 0
