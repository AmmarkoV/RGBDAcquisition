#LAFAN
Hips
        JOINT LeftUpLeg
                JOINT LeftLeg
                        JOINT LeftFoot
                                JOINT LeftToe
        JOINT RightUpLeg
                JOINT RightLeg
                        JOINT RightFoot
                                JOINT RightToe
        JOINT Spine
                JOINT Spine1
                        JOINT Spine2
                                JOINT Neck
                                        JOINT Head
                                JOINT LeftShoulder
                                        JOINT LeftArm
                                                JOINT LeftForeArm
                                                        JOINT LeftHand
                                JOINT RightShoulder
                                        JOINT RightArm
                                                JOINT RightForeArm
                                                        JOINT RightHand
#MINE 

Hip  
  JOINT abdomen
    JOINT chest
      JOINT neck
          JOINT neck1
            JOINT head
              JOINT __jaw
                JOINT jaw
                  JOINT special04
                    JOINT oris02
                      JOINT oris01
                    JOINT oris06.l
                      JOINT oris07.l
                    JOINT oris06.r
                      JOINT oris07.r
                  JOINT tongue00
                    JOINT tongue01
                      JOINT tongue02
                        JOINT tongue03
                          JOINT __tongue04
                            JOINT tongue04
                          JOINT tongue07.l
                          JOINT tongue07.r
                        JOINT tongue06.l
                        JOINT tongue06.r
                      JOINT tongue05.l
                      JOINT tongue05.r
              JOINT __levator02.l
                JOINT levator02.l
                  JOINT levator03.l
                    JOINT levator04.l
                      JOINT levator05.l
              JOINT __levator02.r
                JOINT levator02.r
                  JOINT levator03.r
                    JOINT levator04.r
                      JOINT levator05.r
              JOINT __special01
                JOINT special01
                  JOINT oris04.l
                    JOINT oris03.l
                  JOINT oris04.r
                    JOINT oris03.r
                  JOINT oris06
                    JOINT oris05
              JOINT __special03
                JOINT special03
                  JOINT __levator06.l
                    JOINT levator06.l
                  JOINT __levator06.r
                    JOINT levator06.r
              JOINT special06.l
                JOINT special05.l
                  JOINT eye.l
                  JOINT orbicularis03.l
                  JOINT orbicularis04.l
              JOINT special06.r
                JOINT special05.r
                  JOINT eye.r
                  JOINT orbicularis03.r
                  JOINT orbicularis04.r
              JOINT __temporalis01.l
                JOINT temporalis01.l
                  JOINT oculi02.l
                    JOINT oculi01.l
              JOINT __temporalis01.r
                JOINT temporalis01.r
                  JOINT oculi02.r
                    JOINT oculi01.r
              JOINT __temporalis02.l
                JOINT temporalis02.l
                  JOINT risorius02.l
                    JOINT risorius03.l
              JOINT __temporalis02.r
                JOINT temporalis02.r
                  JOINT risorius02.r
                    JOINT risorius03.r
      JOINT rCollar
        JOINT rShldr
          JOINT rForeArm
            JOINT rHand
                  JOINT metacarpal1.r
                    JOINT finger2-1.r
                      JOINT finger2-2.r
                        JOINT finger2-3.r
                  JOINT metacarpal2.r
                    JOINT finger3-1.r
                      JOINT finger3-2.r
                        JOINT finger3-3.r
                  JOINT __metacarpal3.r
                    JOINT metacarpal3.r
                      JOINT finger4-1.r
                        JOINT finger4-2.r
                          JOINT finger4-3.r
                  JOINT __metacarpal4.r
                    JOINT metacarpal4.r
                      JOINT finger5-1.r
                        JOINT finger5-2.r
                          JOINT finger5-3.r
                  JOINT __rthumb
                    JOINT rthumb
                      JOINT finger1-2.r
                        JOINT finger1-3.r
      JOINT lCollar
        JOINT lShldr
          JOINT lForeArm
            JOINT lHand
                  JOINT metacarpal1.l
                    JOINT finger2-1.l
                      JOINT finger2-2.l
                        JOINT finger2-3.l
                  JOINT metacarpal2.l
                    JOINT finger3-1.l
                      JOINT finger3-2.l
                        JOINT finger3-3.l
                  JOINT __metacarpal3.l
                    JOINT metacarpal3.l
                      JOINT finger4-1.l
                        JOINT finger4-2.l
                          JOINT finger4-3.l
                  JOINT __metacarpal4.l
                    JOINT metacarpal4.l
                      JOINT finger5-1.l
                        JOINT finger5-2.l
                          JOINT finger5-3.l
                  JOINT __lthumb
                    JOINT lthumb
                      JOINT finger1-2.l
                        JOINT finger1-3.l
  JOINT rButtock
    JOINT rThigh
      JOINT rShin
        JOINT rFoot
                                                        JOINT toe1-1.R
                                                                JOINT toe1-2.R
                                                        JOINT toe2-1.R
                                                                JOINT toe2-2.R
                                                                        JOINT toe2-3.R
                                                        JOINT toe3-1.R
                                                                JOINT toe3-2.R
                                                                        JOINT toe3-3.R
                                                        JOINT toe4-1.R
                                                                JOINT toe4-2.R
                                                                        JOINT toe4-3.R
                                                        JOINT toe5-1.R
                                                                JOINT toe5-2.R
                                                                        JOINT toe5-3.R
  JOINT lButtock
    JOINT lThigh
      JOINT lShin
        JOINT lFoot
                                                        JOINT toe1-1.L
                                                                JOINT toe1-2.L
                                                        JOINT toe2-1.L
                                                                JOINT toe2-2.L
                                                                        JOINT toe2-3.L
                                                        JOINT toe3-1.L
                                                                JOINT toe3-2.L
                                                                        JOINT toe3-3.L
                                                        JOINT toe4-1.L
                                                                JOINT toe4-2.L
                                                                        JOINT toe4-3.L
                                                        JOINT toe5-1.L
                                                                JOINT toe5-2.L
                                                                        JOINT toe5-3.L

############################################
JOINT_ASSOCIATION(hip,Hips)
JOINT_ASSOCIATION(abdomen,Spine1)
JOINT_ASSOCIATION(chest,Spine2)
JOINT_ASSOCIATION(neck,Neck)
JOINT_ASSOCIATION(neck1,Head)
JOINT_ASSOCIATION(rCollar,RightShoulder)
JOINT_ASSOCIATION(rShldr,RightArm)
JOINT_ASSOCIATION(rForeArm,RightForeArm)
JOINT_ASSOCIATION(rHand,RightHand)
JOINT_ASSOCIATION(lCollar,LeftShoulder)
JOINT_ASSOCIATION(lShldr,LeftArm)
JOINT_ASSOCIATION(lForeArm,LeftForeArm)
JOINT_ASSOCIATION(lHand,LeftHand)
JOINT_ASSOCIATION(rThigh,RightUpLeg)
JOINT_ASSOCIATION(rShin,RightLeg)
JOINT_ASSOCIATION(rFoot,RightFoot)
JOINT_ASSOCIATION(lThigh,LeftUpLeg)
JOINT_ASSOCIATION(lShin,LeftLeg)
JOINT_ASSOCIATION(lFoot,LeftFoot)
############################################



















