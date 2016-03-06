#!/bin/bash
 

WIDTH="1280"
HEIGHT="720"
FPS="10"

WIDTH="752"
HEIGHT="416"
FPS="30"

#./run_viewer.sh -module V4L2STEREO -from /dev/video0,/dev/video1 -resolution $WIDTH $HEIGHT -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -disparitySADWindowSize 25 $@
#./run_viewer.sh -module V4L2STEREO -from /dev/video0,/dev/video1 -resolution $WIDTH $HEIGHT -noDepth $@



#./run_viewer.sh -module TEMPLATE -seek 50 -from calib1athens   -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500   -disparitySADWindowSize 70 -disparityCalibrate 9 13 0.7 /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/newCalibrationOutput2  $@


#  firstGPSCar firstGPSCar2 allimiagpscalibra firstGPSCar2 firstGPSCar3Pan
#./run_viewer.sh -module TEMPLATE -from firstGPSToTheCity -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500   -disparitySADWindowSize 70 -disparitySwapColorFeeds $@

#calibraInternal pameVoltelikocalib pameVolta2 voltaAthens 
#./run_viewer.sh -module TEMPLATE -from  calibraInternal  firstGPSToTheCity -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500   -disparitySADWindowSize 70 -disparityUseCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/newCalibrationOutput $@
# -executeEveryLoop scrot -disparitySwapColorFeeds 
 
#./run_viewer.sh -module V4L2STEREO -from /dev/video0,/dev/video1 -nolocation -resolution 1280 720 -fps 10 -resizeWindow 1920 500 -saveEveryFrame


#./run_viewer.sh -module TEMPLATE -from voltaAmmoudara/manual  -nolocation -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -disparityCalibrate  9 13 0.7 /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/voltaAmmoudara/calibration $@

#calibAmmoudaraA  -disparityUseCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/voltaAmmoudara/calibration/
./run_viewer.sh -module TEMPLATE -from voltaAmmoudara/pameVolta6 voltaAmmoudara/pameVoltelikocalib -seek 100  -noDepth -resizeWindow 1920 500 -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping   -disparitySADWindowSize 70  $@

#-disparityshiftYLeft 22 -maxFrames 500
#-disparityCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/allimiagpscalibra/DisparityMappingCalibration 


#./run_viewer.sh -module TEMPLATE -from oldStereoSet -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -maxFrames 200  -disparitySADWindowSize 25 $@




exit 0
