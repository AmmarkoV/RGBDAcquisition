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

#calibraInternal
#./run_viewer.sh -module TEMPLATE -from pameVoltelikocalib pameVolta2 voltaAthens calibraInternal  firstGPSToTheCity -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500   -disparitySADWindowSize 70 -disparityUseCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/newCalibrationOutput $@
# -executeEveryLoop scrot -disparitySwapColorFeeds 

./run_viewer.sh -module TEMPLATE -from voltaAmmoudara/calibAmmoudaraA -seek 10 -nolocation -maxFrames 250 -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500 -disparityCalibrate  9 13 0.7 /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/voltaAmmoudara/calibration $@

#calibAmmoudaraA
./run_viewer.sh -module TEMPLATE -from voltaAmmoudara/pameVoltelikocalib -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -resizeWindow 1920 500   -disparitySADWindowSize 70 -disparityUseCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/voltaAmmoudara/calibration/ $@

#-disparityshiftYLeft 22 -maxFrames 500
#-disparityCalibration /home/ammar/Documents/Programming/FORTH/input_acquisition/grabbed_frames/allimiagpscalibra/DisparityMappingCalibration 


#./run_viewer.sh -module TEMPLATE -from oldStereoSet -seek 10  -noDepth -processor ../processors/DisparityMapping/libDisparityMapping.so  DisparityMapping -maxFrames 200  -disparitySADWindowSize 25 $@




exit 0
