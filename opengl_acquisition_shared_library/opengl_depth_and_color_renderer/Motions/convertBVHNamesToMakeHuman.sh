#!/bin/bash

TARGET="02_03MH.bvh"
 
sed -i 's/hip/Hips/g' $TARGET
sed -i 's/abdomen/Spine/g' $TARGET
sed -i 's/chest/Spine1/g' $TARGET
sed -i 's/neck/Neck/g' $TARGET
sed -i 's/head/Head/g' $TARGET
sed -i 's/leftEye/eye.L/g' $TARGET
sed -i 's/rightEye/eye.R/g' $TARGET
sed -i 's/rCollar/RightShoulder/g' $TARGET
sed -i 's/rShldr/RightArm/g' $TARGET
sed -i 's/rForeArm/RightForeArm/g' $TARGET
sed -i 's/rHand/RightHand/g' $TARGET
sed -i 's/rThumb1/RThumb/g' $TARGET
sed -i 's/rThumb2/finger1-3.R/g' $TARGET
sed -i 's/rIndex1/finger2-1.R/g' $TARGET
sed -i 's/rIndex2/finger2-1.R/g' $TARGET
sed -i 's/rMid1/finger3-1.R/g' $TARGET
sed -i 's/rMid2/finger3-1.R/g' $TARGET
sed -i 's/rRing1/finger4-1.R/g' $TARGET
sed -i 's/rRing2/finger4-1.R/g' $TARGET
sed -i 's/rPinky1/finger5-1.R/g' $TARGET
sed -i 's/rPinky2/finger5-1.R/g' $TARGET
sed -i 's/lCollar/LeftShoulder/g' $TARGET
sed -i 's/lShldr/LeftArm/g' $TARGET
sed -i 's/lForeArm/LeftForeArm/g' $TARGET
sed -i 's/lHand/LeftHand/g' $TARGET
sed -i 's/lThumb1/LThumb/g' $TARGET
sed -i 's/lThumb2/finger1-3.L/g' $TARGET
sed -i 's/lIndex1/finger2-1.L/g' $TARGET
sed -i 's/lIndex2/finger2-3.L/g' $TARGET
sed -i 's/lMid1/finger3-1.L/g' $TARGET
sed -i 's/lMid2/finger3-3.L/g' $TARGET
sed -i 's/lRing1/finger4-1.L/g' $TARGET
sed -i 's/lRing2/finger4-3.L/g' $TARGET
sed -i 's/lPinky1/finger5-1.L/g' $TARGET
sed -i 's/lPinky2/finger5-3.L/g' $TARGET
sed -i 's/rButtock/LHip/g' $TARGET
sed -i 's/rThigh/LeftUpLeg/g' $TARGET
sed -i 's/rShin/LeftLeg/g' $TARGET
sed -i 's/rFoot/LeftFoot/g' $TARGET
sed -i 's/lButtock/RHip/g' $TARGET
sed -i 's/lThigh/RightUpLeg/g' $TARGET
sed -i 's/lShin/RightLeg/g' $TARGET
sed -i 's/lFoot/RightFoot/g' $TARGET
  

