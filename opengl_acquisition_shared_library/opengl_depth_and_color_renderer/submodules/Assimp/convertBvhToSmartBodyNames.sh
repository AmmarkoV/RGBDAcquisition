#!/bin/bash

THEFILE="bvh.conf" 
#sed -i 's/Hips/JtHipRt/g' $THEFILE
#sed -i 's/spine/JtSpineA/g' $THEFILE
#sed -i 's/spine1/JtSpineB/g' $THEFILE
#sed -i 's/spine2/JtSpineC/g' $THEFILE
sed -i 's/Neck/JtNeckA/g' $THEFILE
sed -i 's/Head/JtNeckB/g' $THEFILE

sed -i 's/RightShoulder/JtShoulderRt/g' $THEFILE
sed -i 's/RightArm/JtElbowRt/g' $THEFILE
sed -i 's/RightHand/JtWristRt/g' $THEFILE
#sed -i 's/RightArmRoll/
#sed -i 's/RightForeArm/
#sed -i 's/RightForeArmRoll/

sed -i 's/LeftShoulder/JtShoulderLf/g' $THEFILE
sed -i 's/LeftArm/JtElbowLf/g' $THEFILE
sed -i 's/LeftHand/JtWristLf/g' $THEFILE
#sed -i 's/LeftArmRoll/
#sed -i 's/LeftForeArm/
#sed -i 's/LeftForeArmRoll/

sed -i 's/RightUpLeg/JtHipRt/g' $THEFILE
#sed -i 's/RightUpLegRoll/
sed -i 's/RightLeg/JtKneeRt/g' $THEFILE
#sed -i 's/RightLegRoll/
sed -i 's/RightFoot/JtAnkleRt/g' $THEFILE
sed -i 's/RightToeBase/JtToeRt/g' $THEFILE

sed -i 's/LeftUpLeg/JtHipLf/g' $THEFILE
#sed -i 's/LeftUpLegRoll/
sed -i 's/LeftLeg/JtKneeLf/g' $THEFILE
#sed -i 's/LeftLegRoll/
sed -i 's/LeftFoot/JtAnkleLf/g' $THEFILE
sed -i 's/LeftToeBase/JtToeLf/g' $THEFILE
 






exit 0
