#!/bin/bash

FILE="$1"
sed -i 's/hip/Hips/g' $FILE
#--------------------------------------------
sed -i 's/lbuttock/LHipJoint/g' $FILE
sed -i 's/lhip/LeftUpLeg/g' $FILE
sed -i 's/lknee/LeftLeg/g' $FILE
sed -i 's/lfoot/LeftFoot/g' $FILE
sed -i 's/toe2-1.l/LeftToeBase/g' $FILE
#--------------------------------------------
sed -i 's/rbuttock/RHipJoint/g' $FILE
sed -i 's/rhip/RightUpLeg/g' $FILE
sed -i 's/rknee/RightLeg/g' $FILE
sed -i 's/rfoot/RightFoot/g' $FILE
sed -i 's/toe2-1.r/RightToeBase/g' $FILE
#--------------------------------------------
sed -i 's/abdomen/Spine/g' $FILE
sed -i 's/chest/Spine1/g' $FILE
sed -i 's/neck/Neck/g' $FILE
sed -i 's/neck1/Neck1/g' $FILE
sed -i 's/head/Head/g' $FILE
#------------------------------------------------------
sed -i 's/lcollar/LeftShoulder/g' $FILE
sed -i 's/lshoulder/LeftArm/g' $FILE
sed -i 's/lelbow/LeftForeArm/g' $FILE
sed -i 's/lhand/LeftHand/g' $FILE
sed -i 's/metacarpal1.l/LeftFingerBase/g' $FILE
sed -i 's/finger2-1.l/LeftHandIndex1/g' $FILE
sed -i 's/lthumb/LThumb/g' $FILE
#------------------------------------------------------
sed -i 's/rcollar/RightShoulder/g' $FILE
sed -i 's/rshoulder/RightArm/g' $FILE
sed -i 's/relbow/RightForeArm/g' $FILE
sed -i 's/rhand/RightHand/g' $FILE 
sed -i 's/metacarpal1.r/RightFingerBase/g' $FILE
sed -i 's/finger2-1.r/RightHandIndex1/g' $FILE 
sed -i 's/rthumb/RThumb/g' $FILE


exit 0
