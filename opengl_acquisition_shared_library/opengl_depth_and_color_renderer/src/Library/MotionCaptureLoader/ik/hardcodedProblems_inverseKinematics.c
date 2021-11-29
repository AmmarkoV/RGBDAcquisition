#include "hardcodedProblems_inverseKinematics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "../calculate/bvh_transform.h"



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//Attempt to add a second foot solution chain
#define DUALFOOT 1

//Attempt to add a second thumb solution chain
#define DUALTHUMB 1

int addNewPartToChainProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    //-----------------------------------------
    char * partName,
    char * alternatePartName,
    float importance,
    int isEndEffector,
    //-----------------------------------------
    unsigned int * groupID,
    unsigned int * jobID,
    unsigned int * chainID,
    unsigned int * partID,
    //-----------------------------------------
    char forceSpecificMIDs,
    BVHMotionChannelID mIDStart,
    BVHMotionChannelID mIDEnd
    )
{
    if (*chainID >= MAXIMUM_CHAINS)
    {
      fprintf(stderr,RED "Reached limit of maximum chains.. (%d) \n" NORMAL,MAXIMUM_CHAINS);
      return 0;
    }

    if (*partID >= MAXIMUM_PARTS_OF_CHAIN)
    {
      fprintf(stderr,RED "Reached limit of maximum parts of the chain.. (%d) \n" NORMAL,MAXIMUM_PARTS_OF_CHAIN);
      return 0;
    }


    //Chain 0 is the RHand and all of the rigid torso
    //----------------------------------------------------------
    problem->chain[*chainID].groupID=*groupID;
    problem->chain[*chainID].jobID=*jobID;
    problem->chain[*chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);
    problem->chain[*chainID].status = BVH_IK_NOTSTARTED;
    problem->chain[*chainID].permissionToStart = 0;
    problem->chain[*chainID].parallel=0;

    if (!bvh_allocateTransform(mc,&problem->chain[*chainID].current2DProjectionTransform))
    {
      fprintf(stderr,RED "Could not allocate transforms needed for chain (%u/%d) \n" NORMAL,*chainID,MAXIMUM_PARTS_OF_CHAIN);
      return 0;
    }

    bvh_markAllJointsAsUselessInTransform(mc,&problem->chain[*chainID].current2DProjectionTransform);

    unsigned int thisJID=0;

    unsigned int foundJoint = bvh_getJointIDFromJointNameNocase(mc,partName,&thisJID);
    if  ( (!foundJoint) && (alternatePartName!=0) )
    {
        foundJoint = bvh_getJointIDFromJointNameNocase(mc,alternatePartName,&thisJID);
    }


    if (foundJoint)
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[*chainID].current2DProjectionTransform,thisJID);
        problem->chain[*chainID].part[*partID].limits=0;
        problem->chain[*chainID].part[*partID].evaluated=0; //Not evaluated yet
        problem->chain[*chainID].part[*partID].jID=thisJID;
        problem->chain[*chainID].part[*partID].jointImportance=importance;
        problem->chain[*chainID].part[*partID].endEffector=isEndEffector;

        if (isEndEffector)
        {
            //End Effectors do not have/need motion channels..
            problem->chain[*chainID].part[*partID].mIDStart=0;
            problem->chain[*chainID].part[*partID].mIDEnd=0;
        }
         else
        if (!forceSpecificMIDs)
        {
         BVHMotionChannelID mIDAutoStart = mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation encountered
         BVHMotionChannelID mIDAutoEnd = mIDAutoStart + mc->jointHierarchy[thisJID].loadedChannels-1;
         //-------------------------------------------------------------------------------------------------------------------------
         if ( (mIDStart > problem->mc->numberOfValuesPerFrame) || (mIDEnd > problem->mc->numberOfValuesPerFrame) )
             {
                 fprintf(stderr,RED "Bug detected on joint %s (chain id %u / part id %u), coordinates out of mID limits..\n" NORMAL,partName,*chainID,*partID);
                 return 0;
             }
         problem->chain[*chainID].part[*partID].mIDStart=mIDAutoStart;
         problem->chain[*chainID].part[*partID].mIDEnd=mIDAutoEnd;
         unsigned int coordinatesToRegress = 1 + mIDAutoEnd - mIDAutoStart;

         if ( ( coordinatesToRegress != 3) && ( coordinatesToRegress != 0) )
         {
           fprintf(stderr,RED "Bug detected on joint %s (chain id %u / part id %u), coordinates to regress are != 3 and not zero..\n" NORMAL,partName,*chainID,*partID);
           fprintf(stderr,RED "Automatically assigned mID Start: %u\n" NORMAL,problem->chain[*chainID].part[*partID].mIDStart);
           fprintf(stderr,RED "Automatically assigned mID End: %u\n" NORMAL,problem->chain[*chainID].part[*partID].mIDEnd);
         }
        } else
        {
         //Use custom mIDS Start/End ( for root position rotation )
         problem->chain[*chainID].part[*partID].mIDStart=mIDStart;
         problem->chain[*chainID].part[*partID].mIDEnd=mIDEnd;
        }

        *partID+=1;
        problem->chain[*chainID].numberOfParts=*partID;
        //------
        return 1;
    }
    else
    {
        bvh_printBVH(mc);
        fprintf(stderr,RED "No %s in armature..\n" NORMAL,partName);
        if (alternatePartName!=0)
        {
         fprintf(stderr,RED "Also checked for the alternate %s name in armature..\n" NORMAL,alternatePartName);
        }
        //------
        //exit(0); <- this is extreme..
        return 0;
    }
}




int addLimitsToPartOfChain(
                           struct ikProblem * problem,
                           struct BVH_MotionCapture * mc,
                           //-----------------------------------------
                           unsigned int chainID,
                           unsigned int partID,
                           //-----------------------------------------
                           float minimumX,
                           float maximumX,
                           float minimumY,
                           float maximumY,
                           float minimumZ,
                           float maximumZ
                          )
{
    if (chainID >= MAXIMUM_CHAINS)
    {
      fprintf(stderr,RED "Reached limit of maximum chains.. (%d) \n" NORMAL,MAXIMUM_CHAINS);
      return 0;
    }

    if (partID >= MAXIMUM_PARTS_OF_CHAIN)
    {
      fprintf(stderr,RED "Reached limit of maximum parts of the chain.. (%d) \n" NORMAL,MAXIMUM_PARTS_OF_CHAIN);
      return 0;
    }

    problem->chain[chainID].part[partID].limits=1;
    //Z X Y
    problem->chain[chainID].part[partID].minimumLimitMID[0]=minimumZ;
    problem->chain[chainID].part[partID].maximumLimitMID[0]=maximumZ;
    problem->chain[chainID].part[partID].minimumLimitMID[1]=minimumX;
    problem->chain[chainID].part[partID].maximumLimitMID[1]=maximumX;
    problem->chain[chainID].part[partID].minimumLimitMID[2]=minimumY;
    problem->chain[chainID].part[partID].maximumLimitMID[2]=maximumY;
   //------
   return 1;
}





int addEstimatedMAEToPartOfChain(
                                 struct ikProblem * problem,
                                 struct BVH_MotionCapture * mc,
                                 //-----------------------------------------
                                 unsigned int chainID,
                                 unsigned int partID,
                                 //-----------------------------------------
                                 float mAE_X,
                                 float mAE_Y,
                                 float mAE_Z
                                )
{
    if (chainID >= MAXIMUM_CHAINS)
    {
      fprintf(stderr,RED "Reached limit of maximum chains.. (%d) \n" NORMAL,MAXIMUM_CHAINS);
      return 0;
    }

    if (partID >= MAXIMUM_PARTS_OF_CHAIN)
    {
      fprintf(stderr,RED "Reached limit of maximum parts of the chain.. (%d) \n" NORMAL,MAXIMUM_PARTS_OF_CHAIN);
      return 0;
    }

    problem->chain[chainID].part[partID].maeDeclared=1;
    //Z X Y
    problem->chain[chainID].part[partID].mAE[0]=mAE_Z;
    problem->chain[chainID].part[partID].mAE[1]=mAE_X;
    problem->chain[chainID].part[partID].mAE[2]=mAE_Y;
   //------
   return 1;
}






int prepareDefaultFaceProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    int standalone
)
{
    if (problem==0)
         {
           fprintf(stderr,"prepareDefaultFaceProblem called without an ikProblem structure\n");
           return 0;
         }

    //Cleanup problem structure..
    memset(problem,0,sizeof(struct ikProblem));

    problem->mc = mc;
    problem->renderer = renderer;

    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,previousSolution);
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

    snprintf(problem->problemDescription,MAXIMUM_PROBLEM_DESCRIPTION,"Face");


    //Chain #0 is Joint Right Hand-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int correct=0;
    unsigned int checksum=0;
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    //BVHJointID thisJID=0;
    //----------------------------------------------------------


     //Next chain is the Head
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;
     /*
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "chest",0,// Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             */


     if (!standalone)
     {
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck1",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     }


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "head",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "special04",0,// Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.l",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.r",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------








     //Next chain is the R Eye Socket
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "orbicularis03.r",0,  // Top eyelid
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_orbicularis03.r",0,  // Top eyelid
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "orbicularis04.r",0,  // Bottom eyelid
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_orbicularis04.r",0,  // Bottom eyelid
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the L Eye Socket
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "orbicularis03.l",0,  // Top eyelid
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_orbicularis03.l",0,  // Top eyelid
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "orbicularis04.l",0,  // Bottom eyelid
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_orbicularis04.l",0,  // Bottom eyelid
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the R Eye
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.r",0,  // Eye control
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_eye.r",0,  // Eye projection
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the L Eye
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.l",0,  // Eye control
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_eye.l",0,  // Eye projection
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the Mouth
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "jaw",0,  // Bottom mouth/center
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris01",0,  // Bottom mouth/center
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris07.r",0,  // Bottom mouth/right
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris07.l",0,  // Bottom mouth/left
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris05",0,  // Top mouth/center
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris03.r",0,  // Top mouth/right
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "oris03.l",0,  // Top mouth/left
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------




     //Next chain is the Mouth
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "risorius03.l",0,  // Left Cheek middle
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "levator05.l",0,  // Left Cheek middle
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "risorius03.r",0,  // Right Cheek middle
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "levator05.r",0,  // Left Cheek middle
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------




    problem->numberOfChains = chainID;
    //problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;

     return 1;
}






int prepareDefaultRightHandProblem(
                                   struct ikProblem * problem,
                                   struct BVH_MotionCapture * mc,
                                   struct simpleRenderer *renderer,
                                   struct MotionBuffer * previousSolution,
                                   struct MotionBuffer * solution,
                                   struct BVH_Transform * bvhTargetTransform,
                                   int standalone
                                  )
{
    if (problem==0)
         {
           fprintf(stderr,"prepareDefaultRightHandProblem called without an ikProblem structure\n");
           return 0;
         }
    //Cleanup problem structure..
    memset(problem,0,sizeof(struct ikProblem));

    problem->mc = mc;
    problem->renderer = renderer;

    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,previousSolution);
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

    snprintf(problem->problemDescription,MAXIMUM_PROBLEM_DESCRIPTION,"Right Hand");


    //Chain #0 is Joint Right Hand-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int correct=0;
    unsigned int checksum=0;
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    //BVHJointID thisJID=0;
    //----------------------------------------------------------


     if (!standalone)
     {
       //Two modes, if we use a body then we will rely on its positional soluton
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0;
       correct=0;
       partID=0;

       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rshoulder","rShldr",  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "relbow","rForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                    minX/maxX        minY/maxY        minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -180.0,10.0,      -20.0,20.0,     -60.0,60.0);

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at non-standalone rHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------

      ++chainID;
      ++jobID;
      //----------------------------------------------------------
      //----------------------------------------------------------
      //----------------------------------------------------------



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //We add a specific kinematic chain that will just handle the wrist pose since the pose retreived when concatenating
    //seperate hands and bodies can be difficult to estimate..
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,// Joint
                               1.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                    minX/maxX        minY/maxY        minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -180.0,180.0,      -20.0,20.0,     -60.0,60.0);


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at common non-standalone rHand wrist chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------

      ++chainID;
      ++jobID;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     } //end of non-standalone mode that also has a body..
      else
     {
       //Second mode, assuming no body
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0;
       correct=0;
       partID=0;


       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,    // Joint
                               2.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              0, //We assume the root joint is the first and the X pos has index 0
                              2  //We assume the root joint is the first and the Z pos has index 2
                             );
       problem->chain[chainID].part[partID-1].bigChanges=1; //Big changes
       //problem->chain[chainID].part[partID-1].mIDStart=0; //First Position
       //problem->chain[chainID].part[partID-1].mIDEnd=2; //First Position

       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,    // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              3, //We assume the root joint is the first and the first rotation component has index 3
                              5  //We assume the root joint is the first and the last rotation component has index 5
                             );

       problem->chain[chainID].part[partID-1].smallChanges=1; //Small changes
       //problem->chain[chainID].part[partID-1].mIDStart=3; //First Position
       //problem->chain[chainID].part[partID-1].mIDEnd=5; //First Position

       if(mc->jointHierarchy[0].channelRotationOrder==BVH_ROTATION_ORDER_QWQXQYQZ)
       {
         //Since quaternions have 4 coordinates, and the main loop of optimization only handles 3
         //We add another "chain" to cover everything
         fprintf(stderr,"Initialization of rhand uses quaternion..\n"); //ignore w

         //Add quaternion limit to previous
         //                                                  minQW/maxQW    minQX/maxQX     minQY/maxQY
         addLimitsToPartOfChain(problem,mc,chainID,partID-1, -1.0,1.0,      -1.0,1.0,        -1.0,1.0);
         //                                                         mAE qW    mAE qX    mAE qY
         addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,  0.25,      0.34,     0.25 );

         ++correct;
         checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,    // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              4, //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components starting from 4
                              6  //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components ending at 4
                             );
        problem->chain[chainID].part[partID-1].smallChanges=1; //Small changes
        //problem->chain[chainID].part[partID-1].mIDStart=4;
        //problem->chain[chainID].part[partID-1].mIDEnd=6;
         //                                                  minQX/maxQX    minQY/maxQY     minQZ/maxQZ
         addLimitsToPartOfChain(problem,mc,chainID,partID-1, -1.0,1.0,      -1.0,1.0,     -1.0,1.0);
         //                                                         mAE qX    mAE qY    mAE qZ
         addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,  0.34,      0.25,     0.34 );
       }

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at standalone rHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------

     //The rest is common for both standalone and non standalone hands..!
       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at standalone/non-standalone common rHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------

      ++chainID;
      ++jobID;
      //----------------------------------------------------------
      //----------------------------------------------------------
      //----------------------------------------------------------

     } //End of standalone chain mode





     //CHAIN 1 ----------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     /*
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0, // Joint
                               2.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-15.0,15.0,  -45.0,90.0,   -17.0,45.0);*/

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------


    //Chain 2 is the Finger 2
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                  -------    minY/maxY     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -20.0,20.0,   -10.0,90.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     2.3,    3.2 );

    ++correct;
    checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    0.0,90.0);
    //                                                         mAE X    mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,    13.2 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-3.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    0.0,45.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,      7.8 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger2-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------






    //Chain 3 is the Finger 3
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                  -------    minY/maxY     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -10.0,10.0,   -10.0,90.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     1.3,     13.2 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    0.0,90.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,     13.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-3.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,     0.0,45.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,      8.4 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger3-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 4 is the Finger 4
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -10.0,10.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     1.7,     13.7 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    0.0,90.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,    13.1 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-3.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,     0.0,45.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,      0.0,     8.3 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger4-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------






    //Chain 5 is the Finger 5
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY    minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -25.0,8.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     2.0,     13.7 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    0.0,90.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,      0.0,     13.9 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-3.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,     0.0,45.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,     4.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger5-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------


    #if DUALTHUMB
    //Thumb is complex so it has a minichain
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumbBase","__rthumb", // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                    minX/maxX   minY/maxY    minZ/maxZ
     //addLimitsToPartOfChain(problem,mc,chainID,partID-1,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     //addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   3.0,     2.6,     2.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger1-3.r",0, // Joint
                              3.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    #endif






    //Chain 6 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumbBase","__rthumb", // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                    minX/maxX   minY/maxY    minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   3.0,     2.6,     2.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX    minY/maxY   minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -48.0,30.0,  -85.0,0.0,   -85.0,85.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   5.5,     14.8,     11.1 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger1-2.r",0, // Joint
                              1.5,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -45.0,45.0,  -35.0,70.0,    0.0,35.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   6.7,     6.1,      2.8 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger1-3.r",0, // Joint
                              2.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                    minX/max   -------     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,  -50.0,0.0,  0.0,0.0,    0.0,50.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   9.0,     0.0,      3.9 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger1-3.r",0, // Joint
                              5.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------

    problem->numberOfChains = chainID;
    //problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;

  return 1;
}











int prepareDefaultLeftHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    int standalone
)
{
    if (problem==0)
         {
           fprintf(stderr,"prepareDefaultLeftHandProblem called without an ikProblem structure\n");
           return 0;
         }
    //Cleanup problem structure..
    memset(problem,0,sizeof(struct ikProblem));

    problem->mc = mc;
    problem->renderer = renderer;

    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,previousSolution);
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

    snprintf(problem->problemDescription,MAXIMUM_PROBLEM_DESCRIPTION,"Left Hand");


    //Chain #0 is Joint Right Hand-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int correct=0;
    unsigned int checksum=0;
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    //----------------------------------------------------------



    //IK Tuning settings
    //----------------------------------------------------------
    //{0.2,1.0,2.0,3.0}; Mean 14.1
    //{0.5,1.0,1.5,2.0}; Mean 13.6
    //{0.3,0.6,1.2,2.0}; Mean 13.6231
    //{0.3,0.6,1.2,2.5}; Mean 13.8871
    //{0.3,0.6,1.2,2.0}  Mean 13.67104
    //{0.3,0.6,1.2,2.5}; Mean 13.67104
    //{0.2,0.6,1.2,1.7}; Mean 13.445
    //{0.2,0.5,1.1,1.5}; Mean 13.5842
    //{0.2,0.5,1.1,1.7}; Mean 13.5802 with lr(0.001) -> Mean   :17.9565  lr (0.02) ->  Mean   : 15.424 // Mean   :10.7307
    //{0.1,0.5,1.0,1.5}; Mean 10.7725
    //{0.1,0.5,1.1,1.6}; Mean   :10.50822
    //{0.1,0.5,1.1,1.5}; Mean   :10.5206
    //{0.1,0.5,1.1,1.65}; Mean   :10.5783
    //{0.1,0.5,1.1,1.55}; Mean   :10.4840
    //{0.1,0.5,1.0,1.55}; Mean   :10.5080
    //Switch to 128 / 59B
    //{0.1,0.5,1.15,1.55}; Mean   :10.5741

    //----------------------------------------------------------
     float allTuningInOne[]={0.1,0.5,1.15,1.55};
     float BASE_ENDPOINT_IMPORTANCE     = allTuningInOne[0];
     float CLOSEST_ENDPOINT_IMPORTANCE  = allTuningInOne[1];
     float MEDIAN_ENDPOINT_IMPORTANCE   = allTuningInOne[2];
     float FURTHEST_ENDPOINT_IMPORTANCE = allTuningInOne[3];
    //----------------------------------------------------------


     if (standalone==0)
     {
     //Next chain is the L Shoulder
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lshoulder","lShldr",  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lelbow","lForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                    minX/maxX        minY/maxY        minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -10.0,180.0,      -20.0,20.0,     -60.0,60.0);

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at non-standalone lHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //We add a specific kinematic chain that will just handle the wrist pose since the pose retreived when concatenating
    //seperate hands and bodies can be difficult to estimate..
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,// Joint
                               BASE_ENDPOINT_IMPORTANCE,     //Importance What importance does this joint give in the rotation..
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                    minX/maxX        minY/maxY        minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -180.0,180.0,      -20.0,20.0,     -60.0,60.0);

     } //end of non-standalone mode that also has a body..
      else
     {
       //Second mode, assuming no body
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0;
       correct=0;
       partID=0;

       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,    // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              0, //We have a position which since it comes from root joint should start at 0
                              2  //We have a position which since it comes from root joint should end at 2
                             );
       unsigned int partThatJustWasCreated = partID - 1;
       problem->chain[chainID].part[partThatJustWasCreated].bigChanges=1; //Big changes

       ++correct;
       checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,    // Joint
                               BASE_ENDPOINT_IMPORTANCE,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              3, //We have a rotation which since it comes from root joint should start at 3
                              5  //We have a rotation which since it comes from root joint should end at 5
                             );
       partThatJustWasCreated = partID - 1;
       problem->chain[chainID].part[partThatJustWasCreated].smallChanges=1; //Small changes

       if(mc->jointHierarchy[0].channelRotationOrder==BVH_ROTATION_ORDER_QWQXQYQZ)
       {
         //Since quaternions have 4 coordinates, and the main loop of optimization only handles 3
         //We add another "chain" to cover everything
         fprintf(stderr,"Initialization of lhand uses quaternion..\n"); //ignore w

         //                                                  minQW/maxQW    minQX/maxQX     minQY/maxQY
         addLimitsToPartOfChain(problem,mc,chainID,partID-1, -1.0,1.0,       -1.0,1.0,       -1.0,1.0);
         //                                                         mAE qW    mAE qX    mAE qY
         addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,  0.25,      0.34,     0.25 );

         ++correct;
         checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,    // Joint
                               BASE_ENDPOINT_IMPORTANCE,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              4, //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components starting from 4
                              6  //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components ending at 4
                             );

         partThatJustWasCreated = partID - 1;
         problem->chain[chainID].part[partThatJustWasCreated].smallChanges=1; //Small changes
         //fprintf(stderr,"mIDS Q2 %u -> %u ..\n", problem->chain[chainID].part[partThatJustWasCreated].mIDStart, problem->chain[chainID].part[partThatJustWasCreated].mIDEnd );
         //                                                  minQX/maxQX    minQY/maxQY     minQZ/maxQZ
         addLimitsToPartOfChain(problem,mc,chainID,partID-1, -1.0,1.0,      -1.0,1.0,     -1.0,1.0);
         //                                                         mAE qX    mAE qY    mAE qZ
         addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,  0.34,      0.25,     0.34 );
       }

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at standalone lHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------
    }


    //This is the common bases of fingers
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.l",0, // Joint Priority
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.l",0, // Joint
                              0.9,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.l",0, // Joint
                              0.9,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.l",0, // Joint Priority
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

      //----------------------------------------------------------
      if (correct!=checksum)
         { fprintf(stderr,"Failed at common standalone/non-standalone lHand chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
      //----------------------------------------------------------

      ++chainID;
      ++jobID;
      //----------------------------------------------------------
      //----------------------------------------------------------
      //----------------------------------------------------------




    //Chain 1 is the Finger 2
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-1.l",0,           // Joint
                              BASE_ENDPOINT_IMPORTANCE,  //Importance
                              0,                         //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -20.0,20.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     2.3,    3.2 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-2.l",0,             //Joint
                              CLOSEST_ENDPOINT_IMPORTANCE, //Importance
                              0,                           //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X    mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,    13.2 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger2-3.l",0,            //Joint
                              MEDIAN_ENDPOINT_IMPORTANCE, //Importance
                              0,                          //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,      7.8 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger2-3.l",0,      // Joint
                              FURTHEST_ENDPOINT_IMPORTANCE, //Importance
                              1,                            //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------






    //Chain 2 is the Finger 3
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-1.l",0,          // Joint
                              BASE_ENDPOINT_IMPORTANCE, //Importance
                              0,                        //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -10.0,10.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     1.3,     13.2 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-2.l",0,              //Joint
                              CLOSEST_ENDPOINT_IMPORTANCE,  //Importance
                              0,                            //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X    mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,     13.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger3-3.l",0,            //Joint
                              MEDIAN_ENDPOINT_IMPORTANCE, //Importance
                              0,                          //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,      8.4 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger3-3.l",0,       //Joint
                              FURTHEST_ENDPOINT_IMPORTANCE,  //Importance
                              1,                             //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 3 is the Finger 4
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-1.l",0,           // Joint
                              BASE_ENDPOINT_IMPORTANCE,  //Importance
                              0,                         //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -10.0,10.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     1.7,     13.7 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-2.l",0,              // Joint
                              CLOSEST_ENDPOINT_IMPORTANCE,  //Importance
                              0,                            //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,    13.1 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger4-3.l",0,              // Joint
                              MEDIAN_ENDPOINT_IMPORTANCE,   //Importance
                              0,                            //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,      0.0,     8.3 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger4-3.l",0, // Joint
                              FURTHEST_ENDPOINT_IMPORTANCE,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------






    //Chain 4 is the Finger 5
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-1.l",0,              // Joint
                              BASE_ENDPOINT_IMPORTANCE,     //Importance
                              0,                            //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  -------    minY/maxY      minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,0.0,  -8.0,25.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     2.0,     13.7 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-2.l",0,                 // Joint
                              CLOSEST_ENDPOINT_IMPORTANCE,     //Importance
                              0,                               //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,      0.0,     13.9 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger5-3.l",0,                // Joint
                              MEDIAN_ENDPOINT_IMPORTANCE,     //Importance
                              0,                              //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                   -------   -------     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   0.0,     0.0,     4.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger5-3.l",0, // Joint
                              FURTHEST_ENDPOINT_IMPORTANCE,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------




    #if DUALTHUMB
    //Thumb is complex so it has a minichain
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lthumbBase","__lthumb", // Joint
                              BASE_ENDPOINT_IMPORTANCE,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     problem->chain[chainID].part[partID-1].bigChanges=1;
     //                                                    minX/maxX   minY/maxY    minZ/maxZ
     //addLimitsToPartOfChain(problem,mc,chainID,partID-1,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     //addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   3.0,     2.6,     2.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lthumb",0,                      // Joint
                              CLOSEST_ENDPOINT_IMPORTANCE,     //Importance
                              1,                               //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger1-3.l",0,          // Joint
                              FURTHEST_ENDPOINT_IMPORTANCE,     //Importance
                              1,                                //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    #endif






    //Chain 5 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..

     ++correct;

     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lthumbBase","__lthumb",                // Joint
                              BASE_ENDPOINT_IMPORTANCE,               //Importance
                              0,                                      //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX     minY/maxY   minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,  -35.0,0.0,    0.0,60.0,    0.0,60.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   13.0,     20.0,     20.0 );
     //addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   3.0,     2.6,     2.6 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lthumb",0,                     // Joint
                              CLOSEST_ENDPOINT_IMPORTANCE,    //Importance
                              0,                              //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX     minY/maxY    minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,  -30.0,48.0,   0.0,85.0,   -85.0,85.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   15.3,     40.0,     40.0 );
     //addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   5.5,     14.8,     11.1 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger1-2.l",0,               // Joint
                              MEDIAN_ENDPOINT_IMPORTANCE,    //Importance / should this be 0 ?
                              0,                             //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX    minY/maxY      0minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -40.0,45.0,   -70.0,35.0,   -35.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   6.7,     6.1,      2.8 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "finger1-3.l",0,               // Joint
                              MEDIAN_ENDPOINT_IMPORTANCE,    //Importance
                              0,                             //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX    -------     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,  0.0,50.0,    0.0,0.0,    -50.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain(problem,mc,chainID,partID-1,   9.0,     0.0,      3.9 );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_finger1-3.l",0,        // Joint
                              FURTHEST_ENDPOINT_IMPORTANCE,   //Importance
                              1,                              //IsEndEffector
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }

    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------

    problem->numberOfChains = chainID;
    //problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;

  return 1;
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
    if (problem==0)
         {
           fprintf(stderr,"prepareDefaultBodyProblem called without an ikProblem structure\n");
           return 0;
         }

    //Cleanup problem structure..
    memset(problem,0,sizeof(struct ikProblem));


    problem->mc = mc;
    problem->renderer = renderer;

    //problem->penultimateSolution = mallocNewMotionBufferAndCopy(mc,solution);
    problem->previousSolution = mallocNewMotionBufferAndCopy(mc,solution); //previousSolution
    problem->initialSolution  = mallocNewMotionBufferAndCopy(mc,solution);
    problem->currentSolution  = mallocNewMotionBufferAndCopy(mc,solution);

    //2D Projections Targeted
    //----------------------------------------------------------
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

    snprintf(problem->problemDescription,MAXIMUM_PROBLEM_DESCRIPTION,"Body");


    //Chain #0 is Joint Hip-> to all its children
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    unsigned int checksum=0;
    unsigned int correct=0;
    unsigned int groupID=0;
    unsigned int jobID=0;
    unsigned int chainID=0;
    unsigned int partID=0;
    //BVHJointID thisJID=0;
    //----------------------------------------------------------







     //First chain is the Hip and all of the rigid torso
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "hip","Hips",    // Joint
                               2.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              0, //We have a position which since it comes from root joint should start at 0
                              2  //We have a position which since it comes from root joint should end at 2
                             );
     problem->chain[chainID].part[partID-1].bigChanges=1; //Big changes
     //problem->chain[chainID].part[partID-1].mIDStart=0; //First Position
     //problem->chain[chainID].part[partID-1].mIDEnd=2; //First Position

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "hip","Hips",    // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              3, //We have a rotation which since it comes from root joint should start at 3
                              5  //We have a rotation which since it comes from root joint should end at 5
                             );
     //problem->chain[chainID].part[partID-1].mIDStart=3; //First Position
     //problem->chain[chainID].part[partID-1].mIDEnd=5; //First Position

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck",0,    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "head",0,    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rshoulder","rShldr", // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lshoulder","lShldr",      // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhip","rThigh", // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhip","lThigh", // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.l",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.r",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------








     //Next chain is the Chest
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

    /*
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "abdomen",0,// Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,45.0,  -45.0,45.0,   -15.0,15.0);
     */


    /* //Unfortunately there is no Chest joint in the Body25 and trying to solve for one introduces wobblyness..
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "chest",0,// Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,45.0,  -45.0,45.0,   -15.0,15.0);
    */

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck",0, // Joint
                               0.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rshoulder","rShldr", // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lshoulder","lForeArm",    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------


    //These are first group..
    ++groupID;





     //Next chain is the Head
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;
     /*
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "chest",0,// Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             */


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-3.0,11.0,  -22.0,22.0,   -5.0,5.0);

     /*
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "neck1",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-3.0,11.0,  -22.0,22.0,   -5.0,5.0);
     */

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "head",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-7.0,22.0,  -45.0,45.0,   -10.0,10.0);


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.l",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "eye.r",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "ear.l","__temporalis02.l",// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "ear.r","__temporalis02.r",// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------







     //Next chain is the R Shoulder
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rshoulder","rShldr",  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "relbow","rForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                     minX/maxX     minY/maxY     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  -30.0,30.0,    -60.0,60.0,   -170.0,0.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );


    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------





     //Next chain  is the L Shoulder
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lshoulder","lShldr",   // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lelbow","lForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //                                                     minX/maxX     minY/maxY     minZ/maxZ
    addLimitsToPartOfChain(problem,mc,chainID,partID-1,  -30.0,30.0,    -60.0,60.0,   0.0,170.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",0,  // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------










     //Next chain is the Right Foot
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rhip","rThigh", // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -135.0,45.0,  -40.0,50.0,   -10.0,80.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rknee","rShin", // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,135.0,    0.0,0.0,    -10.0,10.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rfoot",0,  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,38.0,    0.0,0.0,    -45.0,45.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe1-2.r",0,  // Big Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe5-3.r",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------








     #if DUALFOOT

     //Next chain is the Right Sole
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "rfoot",0,  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,38.0,    0.0,0.0,    -45.0,45.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe1-2.r",0,  // Big Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe5-3.r",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    #endif












     //Next chain  is the Left Foot
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lhip","lThigh",  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                   minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, -135.0,45.0,  -50.0,40.0,   -80.0,10.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lknee","lShin",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1, 0.0,135.0,    0.0,0.0,    -10.0,10.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lfoot",0,// Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,38.0,    0.0,0.0,    -45.0,45.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe1-2.l",0, // Big Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe5-3.l",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------










     #if DUALFOOT

     //Next chain  is the Left Sole
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0;

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "lfoot",0,// Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
     //                                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToPartOfChain(problem,mc,chainID,partID-1,-10.0,38.0,    0.0,0.0,    -45.0,45.0);

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe1-2.l",0, // Big Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );

     #if DUALFOOT
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,
                              //-----------------------------------------
                              "endsite_toe5-3.l",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID,
                              //-----------------------------------------
                              0,0,0 //Automatic mID Start/End assignment
                             );
    #endif
    //----------------------------------------------------------
    if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
    //----------------------------------------------------------

    problem->chain[chainID].parallel=1; //This has to be done after adding parts / Limbs can be solved in parallel
    ++chainID;
    ++jobID;
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
   #endif








    ++groupID;

    problem->numberOfChains = chainID;
    //problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;

    fprintf(stderr,"Body Problem : \n");
    viewProblem(problem);

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
    unsigned int fIDTarget,
    unsigned int multiThreaded
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
    struct MotionBuffer * penultimateSolution= mallocNewMotionBuffer(mc);



    if ( (solution!=0) && (groundTruth!=0) && (initialSolution!=0) && (previousSolution!=0) )
    {
        if (
            ( bvh_copyMotionFrameToMotionBuffer(mc,initialSolution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,previousSolution,fIDPrevious) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource) ) &&
            ( bvh_copyMotionFrameToMotionBuffer(mc,groundTruth,fIDTarget) )
        )
        {
            //------------------------------------------------------------------------
            if(fIDPrevious>0)
            {
              bvh_copyMotionFrameToMotionBuffer(mc,penultimateSolution,fIDPrevious-1);
            } else
            {
              bvh_copyMotionFrameToMotionBuffer(mc,penultimateSolution,fIDPrevious);
            }
            //------------------------------------------------------------------------
            initialSolution->motion[0]=0;
            initialSolution->motion[1]=0;
            initialSolution->motion[2]=distance;

            previousSolution->motion[0]=0;
            previousSolution->motion[1]=0;
            previousSolution->motion[2]=distance;

            penultimateSolution->motion[0]=0;
            penultimateSolution->motion[1]=0;
            penultimateSolution->motion[2]=distance;

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
                    ikConfig.dontUseSolutionHistory=0;
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
                               bvh_freeTransform(&bvhTargetTransform);
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
                            penultimateSolution,
                            previousSolution,
                            solution,
                            groundTruth,
                            //-------------------
                            &bvhTargetTransform,
                            //-------------------
                            multiThreaded, //Use a single thread unless --mt is supplied before testik command!
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

                    }
                    else
                    {
                        fprintf(stderr,"Failed to run IK code..\n");
                    }


                   //Cleanup allocations needed for the problem..
                   cleanProblem(problem);
                   free(problem);
                }
                else
                {
                    fprintf(stderr,"Could not project 2D points of target..\n");
                }
            }
        }
        freeMotionBuffer(&previousSolution);
        previousSolution=0;
        freeMotionBuffer(&solution);
        solution=0;
        freeMotionBuffer(&initialSolution);
        initialSolution=0;
        freeMotionBuffer(&groundTruth);
        groundTruth=0;
    }

    bvh_freeTransform(&bvhTargetTransform);
    return result;
}




