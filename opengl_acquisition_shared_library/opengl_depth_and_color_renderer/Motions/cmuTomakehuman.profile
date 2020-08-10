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






Joint 0 - hip  | Parent 0 - hip 
Joint 1 - abdomen  | Parent 0 - hip 
Joint 2 - chest  | Parent 1 - abdomen 
Joint 3 - neck  | Parent 2 - chest 
Joint 4 - head  | Parent 3 - neck 
Joint 5 - lefteye  | Parent 4 - head 
Joint 6 - end site  | Parent 5 - lefteye 
Joint 7 - righteye  | Parent 4 - head 
Joint 8 - end site  | Parent 7 - righteye 
Joint 9 - rcollar  | Parent 2 - chest 
Joint 10 - rshoulder  | Parent 9 - rcollar 
Joint 11 - relbow  | Parent 10 - rshoulder 
Joint 12 - rhand  | Parent 11 - relbow 
Joint 13 - rthumb1  | Parent 12 - rhand 
Joint 14 - rthumb2  | Parent 13 - rthumb1 
Joint 15 - end site  | Parent 14 - rthumb2 
Joint 16 - rindex1  | Parent 12 - rhand 
Joint 17 - rindex2  | Parent 16 - rindex1 
Joint 18 - end site  | Parent 17 - rindex2 
Joint 19 - rmid1  | Parent 12 - rhand 
Joint 20 - rmid2  | Parent 19 - rmid1 
Joint 21 - end site  | Parent 20 - rmid2 
Joint 22 - rring1  | Parent 12 - rhand 
Joint 23 - rring2  | Parent 22 - rring1 
Joint 24 - end site  | Parent 23 - rring2 
Joint 25 - rpinky1  | Parent 12 - rhand 
Joint 26 - rpinky2  | Parent 25 - rpinky1 
Joint 27 - end site  | Parent 26 - rpinky2 
Joint 28 - lcollar  | Parent 2 - chest 
Joint 29 - lshoulder  | Parent 28 - lcollar 
Joint 30 - lelbow  | Parent 29 - lshoulder 
Joint 31 - lhand  | Parent 30 - lelbow 
Joint 32 - lthumb1  | Parent 31 - lhand 
Joint 33 - lthumb2  | Parent 32 - lthumb1 
Joint 34 - end site  | Parent 33 - lthumb2 
Joint 35 - lindex1  | Parent 31 - lhand 
Joint 36 - lindex2  | Parent 35 - lindex1 
Joint 37 - end site  | Parent 36 - lindex2 
Joint 38 - lmid1  | Parent 31 - lhand 
Joint 39 - lmid2  | Parent 38 - lmid1 
Joint 40 - end site  | Parent 39 - lmid2 
Joint 41 - lring1  | Parent 31 - lhand 
Joint 42 - lring2  | Parent 41 - lring1 
Joint 43 - end site  | Parent 42 - lring2 
Joint 44 - lpinky1  | Parent 31 - lhand 
Joint 45 - lpinky2  | Parent 44 - lpinky1 
Joint 46 - end site  | Parent 45 - lpinky2 
Joint 47 - rbuttock  | Parent 0 - hip 
Joint 48 - rhip  | Parent 47 - rbuttock 
Joint 49 - rknee  | Parent 48 - rhip 
Joint 50 - rfoot  | Parent 49 - rknee 
Joint 51 - end site  | Parent 50 - rfoot 
Joint 52 - lbuttock  | Parent 0 - hip 
Joint 53 - lhip  | Parent 52 - lbuttock 
Joint 54 - lknee  | Parent 53 - lhip 
Joint 55 - lfoot  | Parent 54 - lknee 
Joint 56 - end site  | Parent 55 - lfoot 






#Main Body
JOINT_ASSOCIATION(hip,Hips) 
JOINT_ASSOCIATION(abdomen,Spine)
JOINT_ASSOCIATION(chest,Spine1)
JOINT_ASSOCIATION(neck,Neck)
JOINT_ASSOCIATION(head,Head)
JOINT_ASSOCIATION(lcollar,LeftShoulder)
JOINT_ASSOCIATION(lshoulder,LeftArm)
JOINT_ASSOCIATION(lelbow,LeftForeArm)
JOINT_ASSOCIATION(lhand,LeftHand)
JOINT_ASSOCIATION(lpinky1,finger5-1_L)
JOINT_ASSOCIATION(lpinky2,finger5-2_L)
JOINT_ASSOCIATION(lring1,finger4-1_L)
JOINT_ASSOCIATION(lring2,finger4-2_L)
JOINT_ASSOCIATION(lmid1,finger3-1_L)
JOINT_ASSOCIATION(lmid2,finger3-2_L)
JOINT_ASSOCIATION(lindex1,finger2-1_L)
JOINT_ASSOCIATION(lindex2,finger2-2_L)
JOINT_ASSOCIATION(lthumb1,finger1-2_L)
JOINT_ASSOCIATION(lthumb2,finger1-3_L)
JOINT_ASSOCIATION(rcollar,RightShoulder)
JOINT_ASSOCIATION(rshoulder,RightArm)
JOINT_ASSOCIATION(relbow,RightForeArm)
JOINT_ASSOCIATION(rhand,RightHand)
JOINT_ASSOCIATION(rpinky1,finger5-1_R)
JOINT_ASSOCIATION(rpinky2,finger5-2_R)
JOINT_ASSOCIATION(rring1,finger4-1_R)
JOINT_ASSOCIATION(rring2,finger4-2_R)
JOINT_ASSOCIATION(rmid1,finger3-1_R)
JOINT_ASSOCIATION(rmid2,finger3-2_R)
JOINT_ASSOCIATION(rindex1,finger2-1_R)
JOINT_ASSOCIATION(rindex2,finger2-2_R)
JOINT_ASSOCIATION(rthumb1,finger1-2_R)
JOINT_ASSOCIATION(rthumb2,finger1-3_R)
JOINT_ASSOCIATION(lhip,LeftUpLeg)
JOINT_ASSOCIATION(lknee,LeftLeg)
JOINT_ASSOCIATION(lfoot,LeftFoot)
JOINT_ASSOCIATION(rhip,RightUpLeg)
JOINT_ASSOCIATION(rknee,RightLeg)
JOINT_ASSOCIATION(rfoot,RightFoot)

#JOINT_SIGN(hip,1,1,1)
JOINT_OFFSET(hip,0,0,0)

 

#JOINT_ROTATION_ORDER(lshoulder,z,y,x)
#JOINT_ROTATION_ORDER(rshoulder,z,y,x)
#JOINT_OFFSET(lshoulder,0,45,90)
#JOINT_OFFSET(rshoulder,0,-45,-90) 

#JOINT_ROTATION_ORDER(hip,z,y,x)
#JOINT_ROTATION_ORDER(lelbow,z,x,y)
#JOINT_ROTATION_ORDER(relbow,z,x,y)
#JOINT_OFFSET(lelbow,0,0,-90)
#JOINT_OFFSET(relbow,0,0,90)

#JOINT_SIGN(lelbow,-1,-1,-1)
#JOINT_SIGN(relbow,-1,-1,-1)
#JOINT_SIGN(lshoulder,-1,-1,-1)
#JOINT_SIGN(rshoulder,-1,-1,-1)


#JOINT_SIGN(lknee,-1,-1,-1)
#JOINT_SIGN(rknee,-1,-1,-1)
#JOINT_SIGN(lhip,-1,-1,-1)
#JOINT_SIGN(rhip,-1,-1,-1)



