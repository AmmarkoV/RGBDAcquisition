
#Datasets 01-19
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

cat head.bvh | grep JOINT

ROOT head
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


#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------

#This is basically the same with just added joints so it is very easy to merge

JOINT_ASSOCIATION_SAME(hip)
JOINT_ASSOCIATION_SAME(abdomen)
JOINT_ASSOCIATION_SAME(chest)
JOINT_ASSOCIATION_SAME(neck)
JOINT_ASSOCIATION_SAME(head)
JOINT_ASSOCIATION(lefteye,eye.l)
JOINT_ASSOCIATION(righteye,eye.r)
JOINT_ASSOCIATION_SAME(rcollar)
JOINT_ASSOCIATION_SAME(rshoulder)
JOINT_ASSOCIATION_SAME(relbow)
JOINT_ASSOCIATION_SAME(rhand)
JOINT_ASSOCIATION_SAME(rthumb1)
JOINT_ASSOCIATION_SAME(rthumb2)
JOINT_ASSOCIATION_SAME(rindex1)
JOINT_ASSOCIATION_SAME(rindex2)
JOINT_ASSOCIATION_SAME(rmid1)
JOINT_ASSOCIATION_SAME(rmid2)
JOINT_ASSOCIATION_SAME(rring1)
JOINT_ASSOCIATION_SAME(rring2)
JOINT_ASSOCIATION_SAME(rpinky1)
JOINT_ASSOCIATION_SAME(rpinky2)
JOINT_ASSOCIATION_SAME(lcollar)
JOINT_ASSOCIATION_SAME(lshoulder)
JOINT_ASSOCIATION_SAME(lelbow)
JOINT_ASSOCIATION_SAME(lhand)
JOINT_ASSOCIATION_SAME(lthumb1)
JOINT_ASSOCIATION_SAME(lthumb2)
JOINT_ASSOCIATION_SAME(lindex1)
JOINT_ASSOCIATION_SAME(lindex2)
JOINT_ASSOCIATION_SAME(lmid1)
JOINT_ASSOCIATION_SAME(lmid2)
JOINT_ASSOCIATION_SAME(lring1)
JOINT_ASSOCIATION_SAME(lring2)
JOINT_ASSOCIATION_SAME(lpinky1)
JOINT_ASSOCIATION_SAME(lpinky2)
JOINT_ASSOCIATION_SAME(rbuttock)
JOINT_ASSOCIATION_SAME(rhip)
JOINT_ASSOCIATION_SAME(rknee)
JOINT_ASSOCIATION_SAME(rfoot)
JOINT_ASSOCIATION_SAME(lbuttock)
JOINT_ASSOCIATION_SAME(lhip)
JOINT_ASSOCIATION_SAME(lknee)
JOINT_ASSOCIATION_SAME(lfoot)  
