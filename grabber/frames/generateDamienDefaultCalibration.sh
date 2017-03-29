#!/bin/bash
function generateFile {

THEDATETAG=`date +"%y-%m-%d_%H-%M-%S"`
echo "%Default Calibration File for Damien's BodyTrackerLib/DepthMapTools.cpp"
echo "%CameraID=0"
echo "%CameraNo=0"
echo "%Date=$THEDATETAG"
echo "%ImageWidth=640"
echo "%ImageHeight=480"
echo "%Description=After 0 images , board is 0x0 , square size is 0.000000 , aspect ratio 1.33"
echo "%Intrinsics I[1,1], I[1,2], I[1,3], I[2,1], I[2,2], I[2,3], I[3,1], I[3,2] I[3,3] 3x3"
echo "%I"
echo "575.816"
echo "0.000000"
echo "320.000000"
echo "0.000000"
echo "575.816"
echo "240.000000"
echo "0.000000"
echo "0.000000"
echo "1.000000"
echo "%Distortion D[1], D[2], D[3], D[4] D[5]" 
echo "%D"
echo "0.000000"
echo "0.000000"
echo "0.000000"
echo "0.000000"
echo "0.000000"
echo ""
                }  

generateFile >  $1
exit 0
