#include "hardcodedProblems_inverseKinematics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int prepareDefaultLeftHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
)
{
    problem->mc = mc;
    problem->renderer = renderer;

    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,previousSolution);
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;



    //Chain #0 is Joint Right Hand-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    BVHJointID thisJID=0;
    //----------------------------------------------------------





    //Chain 0 is the RHand and all of the rigid torso
    //----------------------------------------------------------
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=0;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);
  
    if (bvh_getJointIDFromJointName(mc,"rhand",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID; 
        problem->chain[chainID].part[partID].jointImportance=2.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rHand in armature..\n");
        return 0;
    }

}



int prepareDefaultBodyProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
)
{
    problem->mc = mc;
    problem->renderer = renderer;

    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,previousSolution);
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;



    //Chain #0 is Joint Hip-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    BVHJointID thisJID=0;
    //----------------------------------------------------------





    //Chain 0 is the Hip and all of the rigid torso
    //----------------------------------------------------------
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=0;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);
 
    if (bvh_getJointIDFromJointName(mc,"hip",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=0; //First Position
        problem->chain[chainID].part[partID].mIDEnd=2; //First Position
        problem->chain[chainID].part[partID].bigChanges=1;
        problem->chain[chainID].part[partID].jointImportance=2.0;
        ++partID;
    } 


    if (bvh_getJointIDFromJointName(mc,"hip",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=3; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=5; //First Rotation
        problem->chain[chainID].part[partID].jointImportance=2.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No hip in armature..\n");
        return 0;
    }

    if (bvh_getJointIDFromJointName(mc,"neck",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
            
        ++partID;
    }
    else
    {
        fprintf(stderr,"No neck in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShldr",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rshoulder in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShldr",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lshoulder in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) )  || (bvh_getJointIDFromJointName(mc,"rThigh",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.5; //Hips are more important to hips upper body can rotate via chest
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rhip in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lThigh",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.5; //Hips are more important to hips upper body can rotate via chest

        ++partID;
    }
    else
    {
        fprintf(stderr,"No lhip in armature..\n");
        return 0;
    }

    problem->chain[chainID].numberOfParts=partID;

    ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------




    //Chain 1 is the Right Arm
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    partID=0;
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=0;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);

    if (bvh_getJointIDFromJointName(mc,"chest",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=0.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rshoulder in armature..\n");
        return 0;
    }

    if (bvh_getJointIDFromJointName(mc,"neck",&thisJID) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].jointImportance=0.5; //Less important because it can be addressed by chest
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rshoulder in armature..\n");
        return 0;
    }


    if ( (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShldr",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].jointImportance=1.0; //Less important because it can be addressed by chest
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rshoulder in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lForeArm",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].jointImportance=1.0; //Less important because it can be addressed by chest
        ++partID;
    }
    else
    {
        fprintf(stderr,"No relbow in armature..\n");
        return 0;
    }


    problem->chain[chainID].numberOfParts=partID;
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------


//These are first group..
    ++groupID;




    //Chain 1 is the Right Arm
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    partID=0;
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=1;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);

    if ( (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShldr",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=0.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rshoulder in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"relbow",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rForeArm",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0; //This is the parent
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No relbow in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"rhand",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rHand",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=1;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rhand in armature..\n");
        return 0;
    }

    problem->chain[chainID].numberOfParts=partID;
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------





    //Chain 2 is the Left Arm
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    partID=0;
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=1;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);

    if ( (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShldr",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=0.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lshoulder in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lelbow",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lForeArm",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lelbow in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lhand",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lHand",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=1;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lhand in armature..\n");
        return 0;
    }

    problem->chain[chainID].numberOfParts=partID;
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------




    //Chain 3 is the Right Leg
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    partID=0;
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=1;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);

    if ( (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rThigh",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=0.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rhip in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"rknee",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShin",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rknee in armature..\n");
        return 0;
    }


    if ( (bvh_getJointIDFromJointName(mc,"rfoot",&thisJID) )  || (bvh_getJointIDFromJointName(mc,"rFoot",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=1;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No rfoot in armature..\n");
        return 0;
    }


    if ( (bvh_getJointIDFromJointName(mc,"EndSite_toe1-2.r",&thisJID) )  || (bvh_getJointIDFromJointNameNocase(mc,"endsite_toe1-2.r",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=2;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        bvh_printBVH(mc);
        fprintf(stderr,"No R toe in armature..\n");
        return 0;
    }



    problem->chain[chainID].numberOfParts=partID;
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 4 is the Left Leg
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    partID=0;
    problem->chain[chainID].groupID=groupID;
    problem->chain[chainID].jobID=jobID;
    problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[chainID].permissionToStart = 0;
    problem->chain[chainID].parallel=1;

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[chainID].current2DProjectionTransform);

    if ( (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lThigh",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=0.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No hip in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lknee",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShin",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=0;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.5;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lknee in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"lfoot",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lFoot",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=1;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=0;
        problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        fprintf(stderr,"No lfoot in armature..\n");
        return 0;
    }

    if ( (bvh_getJointIDFromJointName(mc,"EndSite_toe1-2.l",&thisJID) )  || (bvh_getJointIDFromJointNameNocase(mc,"endsite_toe1-2.l",&thisJID)) )
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[chainID].current2DProjectionTransform,thisJID);
        problem->chain[chainID].part[partID].partParent=2;
        problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
        problem->chain[chainID].part[partID].jID=thisJID;
        problem->chain[chainID].part[partID].endEffector=1;
        problem->chain[chainID].part[partID].jointImportance=1.0;
        ++partID;
    }
    else
    {
        bvh_printBVH(mc);
        fprintf(stderr,"No L toe in armature..\n");
        return 0;
    }



    problem->chain[chainID].numberOfParts=partID;
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    ++groupID;

    problem->numberOfChains = chainID;
    problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;

    return 1;
}




int writeHTML(
    unsigned int fIDSource,
    unsigned int fIDTarget,
    float initialMAEInPixels,
    float finalMAEInPixels,
    float initialMAEInMM,
    float finalMAEInMM,
    int dumpScreenshots
)
{
    if (dumpScreenshots)
    {
        int i=system("convert initial.svg initial.png&");
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }
        i=system("convert target.svg target.png&");
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }
        i=system("convert solution.svg solution.png&");
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }

        FILE * html=fopen("report.html","w");
        if (html!=0)
        {
            fprintf(html,"<html><body><br>\n");
            //------------------------------------------------------------
            fprintf(html,"<table><tr>\n");
            fprintf(html,"<td><img src=\"initial.png\" width=400></td>\n");
            fprintf(html,"<td><img src=\"target.png\" width=400></td>\n");
            fprintf(html,"<td><img src=\"solution.png\" width=400></td>\n");
            fprintf(html,"</tr>\n");
            //------------------------------------------------------------
            fprintf(html,"<tr>\n");
            fprintf(html,"<td align=\"center\">Initial Pose (frame %u)</td>\n",fIDSource);
            fprintf(html,"<td align=\"center\">Target Pose (frame %u)</td>\n",fIDTarget);
            fprintf(html,"<td align=\"center\">Solution</td>\n");
            fprintf(html,"</tr>\n");
            fprintf(html,"</table>\n");
            //------------------------------------------------------------

            fprintf(html,"<br><br><br><br>");
            fprintf(html,"MAE in 2D Pixels went from %0.2f to %0.2f <br>",initialMAEInPixels,finalMAEInPixels);
            fprintf(html,"MAE in 3D mm went from %0.2f to %0.2f <br>",initialMAEInMM*10,finalMAEInMM*10);

            fprintf(html,"</body></html>");
            fclose(html);
            return 1;
        }

    }
    return 0;
}



float convertStartEndTimeFromMicrosecondsToFPSIK(unsigned long startTime, unsigned long endTime)
{
    float timeInMilliseconds =  (float) (endTime-startTime)/1000;
    if (timeInMilliseconds ==0.0)
        {
            timeInMilliseconds=0.00001;    //Take care of division by null..
        }
    return (float) 1000/timeInMilliseconds;
}

// ./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 80 4 130 0.001 5 100 15

int bvhTestIK(
    struct BVH_MotionCapture * mc,
    float lr,
    float spring,
    unsigned int iterations,
    unsigned int epochs,
    unsigned int fIDPrevious,
    unsigned int fIDSource,
    unsigned int fIDTarget
)
{
    int result=0;

    struct BVH_Transform bvhTargetTransform= {0};

    struct simpleRenderer renderer= {0};
    const float distance=-150;
    simpleRendererDefaults( &renderer, 1920, 1080, 582.18394,   582.52915 );// https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
    simpleRendererInitialize(&renderer);

    fprintf(stderr,"BVH file has motion files with %u elements\n",mc->numberOfValuesPerFrame);

    float initialMAEInPixels=0.0,finalMAEInPixels=0.0;
    float initialMAEInMM=0.0,finalMAEInMM=0.0;

    //Load all motion buffers
    struct MotionBuffer * groundTruth     = mallocNewMotionBuffer(mc);
    struct MotionBuffer * initialSolution = mallocNewMotionBuffer(mc);
    struct MotionBuffer * solution        = mallocNewMotionBuffer(mc);
    struct MotionBuffer * previousSolution= mallocNewMotionBuffer(mc);



    if ( (solution!=0) && (groundTruth!=0) && (initialSolution!=0) && (previousSolution!=0) )
    {
        if (
            ( bvh_copyMotionFrameToMotionBuffer(mc,initialSolution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,previousSolution,fIDPrevious) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,groundTruth,fIDTarget) )
        )
        {
            initialSolution->motion[0]=0;
            initialSolution->motion[1]=0;
            initialSolution->motion[2]=distance;

            previousSolution->motion[0]=0;
            previousSolution->motion[1]=0;
            previousSolution->motion[2]=distance;

            solution->motion[0]=0;
            solution->motion[1]=0;
            solution->motion[2]=distance;


            groundTruth->motion[0]=0;
            groundTruth->motion[1]=0;
            groundTruth->motion[2]=distance;


            if ( bvh_loadTransformForMotionBuffer(mc,groundTruth->motion,&bvhTargetTransform,0) )
            {
                if  (bvh_projectTo2D(mc,&bvhTargetTransform,&renderer,0,0))
                {
                    struct ikConfiguration ikConfig= {0};
                    //------------------------------------
                    ikConfig.learningRate = lr;
                    ikConfig.iterations = iterations;
                    ikConfig.epochs = epochs;
                    ikConfig.spring = spring;
                    ikConfig.gradientExplosionThreshold = 50;
                    ikConfig.dumpScreenshots = 1;
                    ikConfig.maximumAcceptableStartingLoss=0.0; // Dont use this
                    ikConfig.verbose = 1;
                    ikConfig.tryMaintainingLocalOptima=1; //Less Jittery but can be stuck at local optima
                    ikConfig.ikVersion = IK_VERSION;
                    //------------------------------------


    
                       struct ikProblem * problem= (struct ikProblem * ) malloc(sizeof(struct ikProblem));
                      if (problem!=0)
                                     { memset(problem,0,sizeof(struct ikProblem)); } else
                                     { fprintf(stderr,"Failed to allocate memory for our IK problem..\n");  return 0; }

                        if (!prepareDefaultBodyProblem(
                                                                                            problem,
                                                                                            mc,
                                                                                            &renderer,
                                                                                            previousSolution,
                                                                                            solution,
                                                                                            &bvhTargetTransform
                                                                                        )
                         )
                        {
                               fprintf(stderr,"Could not prepare the problem for IK solution\n");
                               free(problem);
                               return 0;
                         }


                   unsigned long startTime = GetTickCountMicrosecondsIK();

                    if (
                        approximateBodyFromMotionBufferUsingInverseKinematics(
                            mc,
                            &renderer,
                            problem,
                            &ikConfig,
                            //---------------
                            0, //This is pre-previous solution but we dont want it for our test..
                            previousSolution,
                            solution,
                            groundTruth,
                            //-------------------
                            &bvhTargetTransform,
                            //-------------------
                            0, //Use a single thread ( For Now )!
                            //------------------- 
                            &initialMAEInPixels,
                            &finalMAEInPixels,
                            &initialMAEInMM,
                            &finalMAEInMM
                        )
                    )
                    {
                        //Important :)
                        //=======
                        result=1;
                        //=======
                        unsigned long endTime = GetTickCountMicrosecondsIK();


                        //-------------------------------------------------------------------------------------------------------------
                        //compareMotionBuffers("The problem we want to solve compared to the initial state",initialSolution,groundTruth);
                        //compareMotionBuffers("The solution we proposed compared to ground truth",solution,groundTruth);
                        //-------------------------------------------------------------------------------------------------------------
                        compareTwoMotionBuffers(mc,"Improvement",initialSolution,solution,groundTruth);

                        fprintf(stderr,"MAE in 2D Pixels went from %0.2f to %0.2f \n",initialMAEInPixels,finalMAEInPixels);
                        fprintf(stderr,"MAE in 3D mm went from %0.2f to %0.2f \n",initialMAEInMM*10,finalMAEInMM*10);
                        fprintf(stderr,"Computation time was %lu microseconds ( %0.2f fps )\n",endTime-startTime,convertStartEndTimeFromMicrosecondsToFPSIK(startTime,endTime));
                        


                        //-------------------------------------------------------------------------------------------------
                        writeHTML(
                            fIDSource,
                            fIDTarget,
                            initialMAEInPixels,
                            finalMAEInPixels,
                            initialMAEInMM,
                            finalMAEInMM,
                            ikConfig.dumpScreenshots
                        );
                        //-------------------------------------------------------------------------------------------------
                        
                       //Cleanup allocations needed for the problem..
                       cleanProblem(problem);
                       free(problem); 
                    }
                    else
                    {
                        fprintf(stderr,"Failed to run IK code..\n");
                    }


                }
                else
                {
                    fprintf(stderr,"Could not project 2D points of target..\n");
                }
            }
        }
        freeMotionBuffer(previousSolution);
        freeMotionBuffer(solution);
        freeMotionBuffer(initialSolution);
        freeMotionBuffer(groundTruth);
    }

    return result;
}




