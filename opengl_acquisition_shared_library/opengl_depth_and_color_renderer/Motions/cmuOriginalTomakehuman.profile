#from CMU BVH to MakeHuman

#MakeHuman DAE file
Bone 0 : Scene 
Bone 1 : CMU_compliant_skeleton 
Bone 2 : Hips 
Bone 3 : LHipJoint 
Bone 4 : LeftUpLeg 
Bone 5 : LeftLeg 
Bone 6 : LeftFoot 
Bone 7 : LeftToeBase 
Bone 8 : LowerBack 
Bone 9 : Spine 
Bone 10 : Spine1 
Bone 11 : LeftShoulder 
Bone 12 : LeftArm 
Bone 13 : LeftForeArm 
Bone 14 : LeftHand 
Bone 15 : metacarpal1_L 
Bone 16 : finger2-1_L 
Bone 17 : finger2-2_L 
Bone 18 : finger2-3_L 
Bone 19 : metacarpal2_L 
Bone 20 : finger3-1_L 
Bone 21 : finger3-2_L 
Bone 22 : finger3-3_L 
Bone 23 : metacarpal3_L 
Bone 24 : finger4-1_L 
Bone 25 : finger4-2_L 
Bone 26 : finger4-3_L 
Bone 27 : metacarpal4_L 
Bone 28 : finger5-1_L 
Bone 29 : finger5-2_L 
Bone 30 : finger5-3_L 
Bone 31 : LThumb 
Bone 32 : finger1-2_L 
Bone 33 : finger1-3_L 
Bone 34 : Neck 
Bone 35 : Neck1 
Bone 36 : Head 
Bone 37 : tongue01 
Bone 38 : levator06_L 
Bone 39 : levator06_R 
Bone 40 : temporalis01_L 
Bone 41 : oculi02_L 
Bone 42 : oculi01_L 
Bone 43 : temporalis01_R 
Bone 44 : oculi02_R 
Bone 45 : oculi01_R 
Bone 46 : orbicularis03_L 
Bone 47 : orbicularis03_R 
Bone 48 : orbicularis04_L 
Bone 49 : orbicularis04_R 
Bone 50 : eye_L 
Bone 51 : eye_R 
Bone 52 : RightShoulder 
Bone 53 : RightArm 
Bone 54 : RightForeArm 
Bone 55 : RightHand 
Bone 56 : metacarpal1_R 
Bone 57 : finger2-1_R 
Bone 58 : finger2-2_R 
Bone 59 : finger2-3_R 
Bone 60 : metacarpal2_R 
Bone 61 : finger3-1_R 
Bone 62 : finger3-2_R 
Bone 63 : finger3-3_R 
Bone 64 : metacarpal3_R 
Bone 65 : finger4-1_R 
Bone 66 : finger4-2_R 
Bone 67 : finger4-3_R 
Bone 68 : metacarpal4_R 
Bone 69 : finger5-1_R 
Bone 70 : finger5-2_R 
Bone 71 : finger5-3_R 
Bone 72 : RThumb 
Bone 73 : finger1-2_R 
Bone 74 : finger1-3_R 
Bone 75 : RHipJoint 
Bone 76 : RightUpLeg 
Bone 77 : RightLeg 
Bone 78 : RightFoot 
Bone 79 : RightToeBase 
Bone 80 : makehumantest-baseObject 
Bone 81 : makehumantest-highpolyeyesObject 


#CMU BVH
  JOINT abdomen
    JOINT chest
      JOINT neck
        JOINT head
          JOINT leftEye
          JOINT rightEye
      JOINT rCollar
        JOINT rShldr
          JOINT rForeArm
            JOINT rHand
              JOINT rThumb1
                JOINT rThumb2
              JOINT rIndex1
                JOINT rIndex2
              JOINT rMid1
                JOINT rMid2
              JOINT rRing1
                JOINT rRing2
              JOINT rPinky1
                JOINT rPinky2
      JOINT lCollar
        JOINT lShldr
          JOINT lForeArm
            JOINT lHand
              JOINT lThumb1
                JOINT lThumb2
              JOINT lIndex1
                JOINT lIndex2
              JOINT lMid1
                JOINT lMid2
              JOINT lRing1
                JOINT lRing2
              JOINT lPinky1
                JOINT lPinky2
  JOINT rButtock
    JOINT rThigh
      JOINT rShin
        JOINT rFoot
  JOINT lButtock
    JOINT lThigh
      JOINT lShin
        JOINT lFoot



#Main Body
JOINT_ASSOCIATION(hip,Hips) 
JOINT_ASSOCIATION(abdomen,Spine)
JOINT_ASSOCIATION(chest,Spine1)
JOINT_ASSOCIATION(neck,Neck)
JOINT_ASSOCIATION(head,Head)
JOINT_ASSOCIATION(lCollar,LeftShoulder)
JOINT_ASSOCIATION(lShldr,LeftArm)
JOINT_ASSOCIATION(lForeArm,LeftForeArm)
JOINT_ASSOCIATION(lHand,LeftHand)
JOINT_ASSOCIATION(lPinky1,finger5-1_L)
JOINT_ASSOCIATION(lPinky2,finger5-2_L)
JOINT_ASSOCIATION(lRing1,finger4-1_L)
JOINT_ASSOCIATION(lRing2,finger4-2_L)
JOINT_ASSOCIATION(lMid1,finger3-1_L)
JOINT_ASSOCIATION(lMid2,finger3-2_L)
JOINT_ASSOCIATION(lIndex1,finger2-1_L)
JOINT_ASSOCIATION(lIndex2,finger2-2_L)
JOINT_ASSOCIATION(lThumb1,finger1-1_L)
JOINT_ASSOCIATION(lThumb2,finger1-2_L)
JOINT_ASSOCIATION(rCollar,RightShoulder)
JOINT_ASSOCIATION(rShldr,RightArm)
JOINT_ASSOCIATION(rForeArm,RightForeArm)
JOINT_ASSOCIATION(rHand,RightHand)
JOINT_ASSOCIATION(rPinky1,finger5-1_R)
JOINT_ASSOCIATION(rPinky2,finger5-2_R)
JOINT_ASSOCIATION(rRing1,finger4-1_R)
JOINT_ASSOCIATION(rRing2,finger4-2_R)
JOINT_ASSOCIATION(rMid1,finger3-1_R)
JOINT_ASSOCIATION(rMid2,finger3-2_R)
JOINT_ASSOCIATION(rIndex1,finger2-1_R)
JOINT_ASSOCIATION(rIndex2,finger2-2_R)
JOINT_ASSOCIATION(rThumb1,finger1-1_R)
JOINT_ASSOCIATION(rThumb2,finger1-2_R)
JOINT_ASSOCIATION(lThigh,LeftUpLeg)
JOINT_ASSOCIATION(lShin,LeftLeg)
JOINT_ASSOCIATION(lFoot,LeftFoot)
JOINT_ASSOCIATION(rThigh,RightUpLeg)
JOINT_ASSOCIATION(rShin,RightLeg)
JOINT_ASSOCIATION(rFoot,RightFoot)





