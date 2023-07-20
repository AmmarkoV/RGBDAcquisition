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

#define USE_CHEST 0


// Variable Shortcut
//------------------------------
const int END_EFFECTOR      = 1;
const int OPTIMIZE_JOINT    = 0;
char * NO_ALTERNATE_NAME    = 0;
//------------------------------

struct problemData
{
 struct ikProblem * problem;
 struct BVH_MotionCapture * mc;
 //---------------------------
 unsigned int jobID;
 unsigned int chainID;
 unsigned int partID;
};

int failedPreparingChain(struct problemData * data,int correct,int checksum)
{
  if (correct!=checksum)
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",data->chainID,checksum,correct); return 1; }
  return 0;
}

void expectBigValueChangesForNextPart(struct problemData * data)
{
  if ( (data!=0) && (data->problem!=0) && (data->chainID<MAXIMUM_CHAINS) && (data->partID<MAXIMUM_PARTS_OF_CHAIN) )
  {
     unsigned int chainID = data->chainID;
     unsigned int partID  = data->partID;
     fprintf(stderr,"Asked to signal big changes to chain %u part %u",chainID,partID);
     data->problem->chain[chainID].part[partID].bigChanges=1; //Big changes
  } else
  {
      fprintf(stderr,"ERROR: You need to increase the MAXIMUM_CHAINS declaration from %u to %u",MAXIMUM_CHAINS,data->chainID+1);
      fprintf(stderr,"ERROR: You need to increase the MAXIMUM_PARTS_OF_CHAIN declaration from %u to %u",MAXIMUM_PARTS_OF_CHAIN,data->partID+1);
  }
}

void expectSmallValueChangesForNextPart(struct problemData * data)
{
  if ( (data!=0) && (data->problem!=0) && (data->chainID<MAXIMUM_CHAINS) && (data->partID<MAXIMUM_PARTS_OF_CHAIN) )
  {
   data->problem->chain[data->chainID].part[data->partID].smallChanges=1;
  } else
  {
      fprintf(stderr,"ERROR: You need to increase the MAXIMUM_CHAINS declaration from %u to %u",MAXIMUM_CHAINS,data->chainID+1);
      fprintf(stderr,"ERROR: You need to increase the MAXIMUM_PARTS_OF_CHAIN declaration from %u to %u",MAXIMUM_PARTS_OF_CHAIN,data->partID+1);
  }
}

void thisChainCanBeRunInParallel(struct problemData * data)
{
  if ( (data!=0) && (data->problem!=0) && (data->chainID<MAXIMUM_CHAINS) )
    { data->problem->chain[data->chainID].parallel=1; }
}

void nextChain(struct problemData * data)
{
  data->chainID+=1;
  if ((data->chainID>=MAXIMUM_CHAINS) )
  {
      fprintf(stderr,"ERROR: You need to increase the MAXIMUM_CHAINS declaration from %u to %u",MAXIMUM_CHAINS,data->chainID+1);
  }
}

void nextChainAndJob(struct problemData * data)
{
  data->jobID+=1;
  nextChain(data);
}

void startAddingNewPartsToChain(struct problemData * data)
{
    data->partID=0;
}

int addNewPartToChainProblemDetailed(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    //-----------------------------------------
    char * partName,
    char * alternatePartName,
    float importance,
    int isEndEffector,
    //-----------------------------------------
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
    //-------------------------------------------------------------------------------------------------
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
    //Resolve Joint Name
    //---------------------------------------------------------------------------------
    unsigned int foundJoint = bvh_getJointIDFromJointNameNocase(mc,partName,&thisJID);
    //---------------------------------------------------------------------------------
    if  ( (!foundJoint) && (alternatePartName!=0) )
    {
        foundJoint = bvh_getJointIDFromJointNameNocase(mc,alternatePartName,&thisJID);
    }
    //----------------------------------------------------------------------------------

    if (foundJoint)
    {
        bvh_markJointAndParentsAsUsefulInTransform(mc,&problem->chain[*chainID].current2DProjectionTransform,thisJID);
        //problem->chain[*chainID].part[*partID].limits=0; //<- if this is 0 it erases limits ( .. bug found 16/3/23 :( )
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
         BVHMotionChannelID mIDAutoEnd   = mIDAutoStart + mc->jointHierarchy[thisJID].loadedChannels-1;
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
        return 0;
    }
}

int addNewPartToChainProblem(
    struct problemData * data,
    //-----------------------------------------
    char * partName,
    char * alternatePartName,
    float importance,
    int isEndEffector
    )
{
    return addNewPartToChainProblemDetailed(data->problem,
                                            data->mc,
                                            partName,alternatePartName,importance,isEndEffector,
                                            &data->jobID,
                                            &data->chainID,
                                            &data->partID,
                                            0,0,0);
}


int addLimitsToPartOfChain(
                           struct problemData * data,
                           unsigned int partID,
                           //-----------------------------------------
                           float minimumX,
                           float maximumX,
                           float minimumY,
                           float maximumY,
                           float minimumZ,
                           float maximumZ
                           //-----------------------------------------
                          )
{
    struct ikProblem * problem    = data->problem;
    unsigned int chainID          = data->chainID;

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
    //---------------------------------------------------------------
    problem->chain[chainID].part[partID].minimumLimitMID[0]=minimumZ;
    problem->chain[chainID].part[partID].maximumLimitMID[0]=maximumZ;
    //---------------------------------------------------------------
    problem->chain[chainID].part[partID].minimumLimitMID[1]=minimumX;
    problem->chain[chainID].part[partID].maximumLimitMID[1]=maximumX;
    //---------------------------------------------------------------
    problem->chain[chainID].part[partID].minimumLimitMID[2]=minimumY;
    problem->chain[chainID].part[partID].maximumLimitMID[2]=maximumY;
    //------
    return 1;
}

int addLimitsToNextPartOfChain(struct problemData * data,float minimumX,float maximumX,float minimumY,float maximumY,float minimumZ,float maximumZ)
{
    return addLimitsToPartOfChain(data,data->partID,minimumX,maximumX,minimumY,maximumY,minimumZ,maximumZ);
}

int addLimitsToPreviousPartOfChain(struct problemData * data,float minimumX,float maximumX,float minimumY,float maximumY,float minimumZ,float maximumZ)
{
    return addLimitsToPartOfChain(data,data->partID-1,minimumX,maximumX,minimumY,maximumY,minimumZ,maximumZ);
}

int addEstimatedMAEToPartOfChain(
                                 struct problemData * data,
                                 unsigned int partID,
                                 //-----------------------------------------
                                 float mAE_X,
                                 float mAE_Y,
                                 float mAE_Z
                                )
{
    struct ikProblem * problem    = data->problem;
    unsigned int chainID          = data->chainID;

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

int addEstimatedMAEToPartOfChain_AfterAddingNewPart(struct problemData * data,float mAE_X,float mAE_Y,float mAE_Z)
{
  return addEstimatedMAEToPartOfChain(data,data->partID-1,mAE_X,mAE_Y,mAE_Z);
}

int addEstimatedMAEToPartOfChain_BeforeAddingNewPart(struct problemData * data,float mAE_X,float mAE_Y,float mAE_Z)
{
  return addEstimatedMAEToPartOfChain(data,data->partID,mAE_X,mAE_Y,mAE_Z);
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
    struct problemData data = {0};
    data.problem = problem;
    data.mc      = mc;
    unsigned int correct=0;
    unsigned int checksum=0;
    //----------------------------------------------------------


     //Next chain is the Head
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //++correct; checksum+=addNewPartToChainProblem(&data"chest",NO_ALTERNATE_NAME,0.5,OPTIMIZE_JOINT);
     if (!standalone)
     {
      //++correct;   checksum+=addNewPartToChainProblem(&data,"neck",NO_ALTERNATE_NAME,0.5,OPTIMIZE_JOINT);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"neck1","neck",0.5,OPTIMIZE_JOINT); //If neck1 is not available ( mnet1-mnet3 ) fallback to old neck
     }
     ++correct;  checksum+=addNewPartToChainProblem(&data,"head",NO_ALTERNATE_NAME,0.5,OPTIMIZE_JOINT);
     ++correct;  checksum+=addNewPartToChainProblem(&data,"special04",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     ++correct;  checksum+=addNewPartToChainProblem(&data,"endsite_eye.l","eye.l",2.5,END_EFFECTOR);
     ++correct;  checksum+=addNewPartToChainProblem(&data,"endsite_eye.r","eye.r",2.5,END_EFFECTOR);
    //----------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //----------------------------------------------------------

    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------




     //Next chain is the R Eye Socket
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct; checksum+=addNewPartToChainProblem(&data,"orbicularis03.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     ++correct; checksum+=addNewPartToChainProblem(&data,"endsite_orbicularis03.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     ++correct; checksum+=addNewPartToChainProblem(&data,"orbicularis04.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     ++correct; checksum+=addNewPartToChainProblem(&data,"endsite_orbicularis04.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------

     nextChainAndJob(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------



     //Next chain is the L Eye Socket
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"orbicularis03.l",0,1.0,OPTIMIZE_JOINT);  // Top eyelid
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_orbicularis03.l",0,1.0,END_EFFECTOR); // Top eyelid
     ++correct;   checksum+=addNewPartToChainProblem(&data,"orbicularis04.l",0,1.0,OPTIMIZE_JOINT);// Bottom eyelid
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_orbicularis04.l",0,1.0,END_EFFECTOR); // Bottom eyelid
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
    //----------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //----------------------------------------------------------

    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the R Eye
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.r","eye.r",1.0,OPTIMIZE_JOINT);// Eye control
     thisChainCanBeRunInParallel(&data);  //This has to be done after adding parts Fingers can be solved in parallel
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------

     nextChainAndJob(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------
    //----------------------------------------------------------



     //Next chain is the L Eye
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);

     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.l","eye.l",1.0,OPTIMIZE_JOINT);// Eye control
     thisChainCanBeRunInParallel(&data);  //This has to be done after adding parts Fingers can be solved in parallel
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     nextChainAndJob(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------



     //Next chain is the Mouth
     //----------------------------------------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"jaw",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);       // Bottom mouth/center
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris01",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);    // Bottom mouth/center
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris07.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);  // Bottom mouth/right
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris07.l",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);  // Bottom mouth/left
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris05",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);    // Top mouth/center
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris03.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);  // Top mouth/right
     ++correct;   checksum+=addNewPartToChainProblem(&data,"oris03.l",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);  // Top mouth/left
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------

    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


     //Next chain is the Cheeks
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"risorius03.l",0,1.0,OPTIMIZE_JOINT);// Left Cheek middle
     ++correct;   checksum+=addNewPartToChainProblem(&data,"levator05.l",0,1.0,OPTIMIZE_JOINT);// Left Cheek middle
     ++correct;   checksum+=addNewPartToChainProblem(&data,"risorius03.r",0,1.0,OPTIMIZE_JOINT);// Right Cheek middle
     ++correct;   checksum+=addNewPartToChainProblem(&data,"levator05.r",0,1.0,OPTIMIZE_JOINT); // Right Cheek middle
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------

     nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------

    //Done!
    problem->numberOfChains = data.chainID;
    problem->numberOfJobs = data.jobID;

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
    struct problemData data = {0};
    data.problem = problem;
    data.mc      = mc;
    unsigned int correct=0;
    unsigned int checksum=0;
    //----------------------------------------------------------


     if (!standalone)
     {
       //Two modes, if we use a body then we will rely on its positional soluton
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0; correct=0; startAddingNewPartsToChain(&data);
       //                                                     minX/maxX     minY/maxY     minZ/maxZ
       addLimitsToNextPartOfChain(&data,    -103.6,104.2,  -192.3,194.6,  -194.54,194.91);
       ++correct; checksum+=addNewPartToChainProblem(&data,"rshoulder","rShldr",0.5,OPTIMIZE_JOINT);
       //                                                     minX/maxX      minY/maxY       minZ/maxZ
       addLimitsToNextPartOfChain(&data,   -68.5,8.37,    -110.0,164.0,   -47.34,35.64);
       ++correct; checksum+=addNewPartToChainProblem(&data,"relbow","rForeArm",1.0,OPTIMIZE_JOINT);
       //                                                    minX/maxX        minY/maxY        minZ/maxZ
       addLimitsToNextPartOfChain(&data,  -180.0,10.0,      -20.0,20.0,     -60.0,60.0);
       ++correct; checksum+=addNewPartToChainProblem(&data,"rhand",NO_ALTERNATE_NAME,1.5,END_EFFECTOR);
       //----------------------------------------------------------
       if (failedPreparingChain(&data,correct,checksum)) { return 0; }
       //----------------------------------------------------------
       nextChainAndJob(&data);
       //----------------------------------------------------------
       //----------------------------------------------------------



      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      //We add a specific kinematic chain that will just handle the wrist pose since the pose retrieved when concatenating
      //seperate hands and bodies can be difficult to estimate..
       checksum=0; correct=0; startAddingNewPartsToChain(&data);
       //                                   minX/maxX        minY/maxY        minZ/maxZ
       addLimitsToNextPartOfChain(&data,   -20.0,20.0,      -180.0,90.0,     -30.0,30.0);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"rhand",NO_ALTERNATE_NAME,1.5,OPTIMIZE_JOINT);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.r",0,1.0,END_EFFECTOR);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.r",0,1.0,END_EFFECTOR);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.r",0,1.0,END_EFFECTOR);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.r",0,1.0,END_EFFECTOR);
       ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumb",0,1.0,END_EFFECTOR);
       //----------------------------------------------------------
       if (failedPreparingChain(&data,correct,checksum)) { return 0; }
       //----------------------------------------------------------
       nextChainAndJob(&data);
       //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     } //end of non-standalone mode that also has a body..
      else
     {
       //Second mode, assuming no body
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0; correct=0; startAddingNewPartsToChain(&data);
       expectBigValueChangesForNextPart(&data);
       ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",NO_ALTERNATE_NAME,
                               2.0,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              0, //We assume the root joint is the first and the X pos has index 0
                              2  //We assume the root joint is the first and the Z pos has index 2
                             );

       expectSmallValueChangesForNextPart(&data);
       ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",NO_ALTERNATE_NAME,
                               1.0,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              3, //We assume the root joint is the first and the first rotation component has index 3
                              5  //We assume the root joint is the first and the last rotation component has index 5
                             );

       if(mc->jointHierarchy[0].channelRotationOrder==BVH_ROTATION_ORDER_QWQXQYQZ)
       {
         //Since quaternions have 4 coordinates, and the main loop of optimization only handles 3
         //We add another "chain" to cover everything
         fprintf(stderr,"Initialization of rhand uses quaternion..\n"); //ignore w
         //Add quaternion limit to previous        minQW/maxQW    minQX/maxQX     minQY/maxQY
         addLimitsToPreviousPartOfChain(&data,     -1.0,1.0,      -1.0,1.0,        -1.0,1.0);
         //                                                       mAE qW    mAE qX    mAE qY
         addEstimatedMAEToPartOfChain_AfterAddingNewPart(&data,  0.25,      0.34,     0.25 );

         //                                                  minQX/maxQX    minQY/maxQY     minQZ/maxQZ
         addLimitsToNextPartOfChain(&data,  -1.0,1.0,      -1.0,1.0,     -1.0,1.0);
         //                                                         mAE qX    mAE qY    mAE qZ
         addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,  0.34,      0.25,     0.34 );
         expectSmallValueChangesForNextPart(&data);
         ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "rhand",NO_ALTERNATE_NAME,
                               1.0,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              4, //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components starting from 4
                              6  //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components ending at 4
                             );
       }
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------

      //The rest is common for both standalone and non standalone hands..!
      ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumb",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
      nextChainAndJob(&data);
      //----------------------------------------------------------
      //----------------------------------------------------------
     } //End of standalone chain mode


     //CHAIN 1 ----------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                  minX/maxX    minY/maxY     minZ/maxZ
     //addLimitsToNextPartOfChain(&data,-15.0,15.0,  -45.0,90.0,   -17.0,45.0);
     //++correct;   checksum+=addNewPartToChainProblem(&data,"rhand",0,2.0,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.r",0,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.r",0,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.r",0,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.r",0,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumb",0,1.0,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
     nextChain(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------


    //Chain 2 is the Finger 2
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data); // Reset counter..
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -20.0,20.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     2.3,    3.2 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.r",0,1.0,OPTIMIZE_JOINT);
     //                                 -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    0.0,90.0);
     //                                                         mAE X    mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,    13.2 );
     ++correct; checksum+=addNewPartToChainProblem(&data,"finger2-2.r",0,1.0,OPTIMIZE_JOINT);
     //                                -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    0.0,45.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,      7.8 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-3.r",0,1.0,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger2-3.r",0,1.0,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 3 is the Finger 3
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data); // Reset counter..
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.r",0,1.0,OPTIMIZE_JOINT);
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -10.0,10.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     1.3,     13.2 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-2.r",0,1.0,OPTIMIZE_JOINT);
     //                                -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    0.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,     13.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-3.r",0,1.0,OPTIMIZE_JOINT);
     //                                 -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,     0.0,45.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,      8.4 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger3-3.r",0,1.0,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 4 is the Finger 4
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data); // Reset counter..
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.r",0,1.0,OPTIMIZE_JOINT);
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -10.0,10.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     1.7,     13.7 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-2.r",0,1.0,OPTIMIZE_JOINT);
     //                                 -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    0.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,    13.1 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-3.r",0,1.0,OPTIMIZE_JOINT);
     //                                -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,     0.0,45.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,      0.0,     8.3 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger4-3.r",0,1.0,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


    //Chain 5 is the Finger 5
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data); // Reset counter..
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     //                                -------    minY/maxY    minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -25.0,8.0,   -10.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     2.0,     13.7 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-2.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     //                                 -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    0.0,90.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,      0.0,     13.9 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-3.r",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     //                                 -------   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,     0.0,45.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,     4.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger5-3.r",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data);//This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


    #if DUALTHUMB
    //Thumb is complex so it has a minichain
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data); // Reset counter..
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumbBase","__rthumb",1.0,OPTIMIZE_JOINT);
     //                                    minX/maxX   minY/maxY    minZ/maxZ
     //addLimitsToNextPartOfChain(&data,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     //addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   3.0,     2.6,     2.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumb",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger1-3.r",NO_ALTERNATE_NAME,3.0,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------
    #endif



    //Chain 6 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumbBase","__rthumb",1.0,OPTIMIZE_JOINT);
     //                                  minX/maxX   minY/maxY    minZ/maxZ
     addLimitsToNextPartOfChain(&data,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   3.0,     2.6,     2.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rthumb",NO_ALTERNATE_NAME,1.0,OPTIMIZE_JOINT);
     //                                  minX/maxX    minY/maxY   minZ/maxZ
     addLimitsToNextPartOfChain(&data, -48.0,30.0,  -85.0,0.0,   -85.0,85.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   5.5,     14.8,     11.1 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger1-2.r",NO_ALTERNATE_NAME,1.5,OPTIMIZE_JOINT);
     //                                 minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, -45.0,45.0,  -35.0,70.0,    0.0,35.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   6.7,     6.1,      2.8 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger1-3.r",NO_ALTERNATE_NAME,2.0,OPTIMIZE_JOINT);
     //                                 minX/max   -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -50.0,0.0,  0.0,0.0,    0.0,50.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   9.0,     0.0,      3.9 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger1-3.r",NO_ALTERNATE_NAME,5.0,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------
    problem->numberOfChains = data.chainID;
    problem->numberOfJobs = data.jobID;

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
    struct problemData data = {0};
    data.problem = problem;
    data.mc      = mc;
    unsigned int correct=0;
    unsigned int checksum=0;
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
      checksum=0; correct=0; startAddingNewPartsToChain(&data);
      //                                   minX/maxX     minY/maxY     minZ/maxZ
      addLimitsToNextPartOfChain(&data,  -103.6,104.2,  -192.3,194.6,  -194.54,194.91);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"lshoulder","lShldr",0.5,OPTIMIZE_JOINT);
      //                                   minX/maxX     minY/maxY     minZ/maxZ
      addLimitsToNextPartOfChain(&data,  -68.5,9.5,    -163.7,110.0,   -35.64,47.64);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"lelbow","lForeArm",1.0,OPTIMIZE_JOINT);
      //                                   minX/maxX     minY/maxY     minZ/maxZ
      addLimitsToNextPartOfChain(&data, -10.0,180.0,    -20.0,20.0,     -60.0,60.0);
      ++correct;   checksum+=addNewPartToChainProblem(&data,"lhand",NO_ALTERNATE_NAME,1.5,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
       nextChainAndJob(&data);
     //----------------------------------------------------------



    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //We add a specific kinematic chain that will just handle the wrist pose since the pose retreived when concatenating
    //seperate hands and bodies can be difficult to estimate..
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                 minX/maxX        minY/maxY        minZ/maxZ
     addLimitsToNextPartOfChain(&data, -20.0,20.0,      -90.0,180.0,     -30.0,30.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lhand",NO_ALTERNATE_NAME,BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT); //What importance does this joint give in the rotation..
     } //end of non-standalone mode that also has a body..
      else
     {
       //Second mode, assuming no body
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       //-----------------------------------------------------------------------
       checksum=0; correct=0;

       expectBigValueChangesForNextPart(&data);
       ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",NO_ALTERNATE_NAME,
                               1.0,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              0, //We have a position which since it comes from root joint should start at 0
                              2  //We have a position which since it comes from root joint should end at 2
                             );

       expectSmallValueChangesForNextPart(&data);
       ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",NO_ALTERNATE_NAME,
                               BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              3, //We have a rotation which since it comes from root joint should start at 3
                              5  //We have a rotation which since it comes from root joint should end at 5
                             );

       if(mc->jointHierarchy[0].channelRotationOrder==BVH_ROTATION_ORDER_QWQXQYQZ)
       {
         //Since quaternions have 4 coordinates, and the main loop of optimization only handles 3
         //We add another "chain" to cover everything

         //                                        minQW/maxQW    minQX/maxQX     minQY/maxQY
         addLimitsToPreviousPartOfChain(&data,    -1.0,1.0,       -1.0,1.0,       -1.0,1.0);
         //                                                     mAE qW    mAE qX    mAE qY
         addEstimatedMAEToPartOfChain_AfterAddingNewPart(&data,  0.25,      0.34,     0.25 );
         fprintf(stderr,"Initialization of lhand uses quaternion..\n"); //ignore w
         expectSmallValueChangesForNextPart(&data);
         //                               minQX/maxQX    minQY/maxQY     minQZ/maxQZ
         addLimitsToNextPartOfChain(&data, -1.0,1.0,      -1.0,1.0,     -1.0,1.0);
         //                                                      mAE qX    mAE qY    mAE qZ
         addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,  0.34,      0.25,     0.34 );
         ++correct; checksum+=addNewPartToChainProblemDetailed(
                              problem,mc,
                              //-----------------------------------------
                              "lhand",NO_ALTERNATE_NAME,
                               BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT,
                              //-----------------------------------------
                              &data.jobID,&data.chainID,&data.partID,
                              //-----------------------------------------
                              1, //Force specific mIDStart/mIDEnd
                              4, //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components starting from 4
                              6  //We have a quaternion which doesnt fit the 3 element structure so  we add one more part with the last 3 rotational components ending at 4
                             );
       }

      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
    }

    //This is the common bases of fingers
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.l",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.l",NO_ALTERNATE_NAME,0.9,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.l",NO_ALTERNATE_NAME,0.9,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.l",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lthumb",NO_ALTERNATE_NAME,1.0,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
      nextChainAndJob(&data);
      //----------------------------------------------------------
      //----------------------------------------------------------
      //----------------------------------------------------------




    //Chain 1 is the Finger 2
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -20.0,20.0,   -90.0,10.0);
     //                                                       mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     2.3,    3.2 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-1.l",NO_ALTERNATE_NAME,BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X    mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,    13.2 );
    ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-2.l",NO_ALTERNATE_NAME,CLOSEST_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,      7.8 );
    ++correct;   checksum+=addNewPartToChainProblem(&data,"finger2-3.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger2-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
    //----------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //----------------------------------------------------------
    thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
    nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------






    //Chain 2 is the Finger 3
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -10.0,10.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     1.3,     13.2 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-1.l",NO_ALTERNATE_NAME,BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X    mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,     13.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-2.l",NO_ALTERNATE_NAME,  CLOSEST_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,      8.4 );
    ++correct;   checksum+=addNewPartToChainProblem(&data,"finger3-3.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger3-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
    //----------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //----------------------------------------------------------
    thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
    nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------



    //Chain 3 is the Finger 4
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                -------    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -10.0,10.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     1.7,     13.7 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-1.l",NO_ALTERNATE_NAME,BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,    13.1 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-2.l",NO_ALTERNATE_NAME,CLOSEST_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,      0.0,     8.3 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger4-3.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger4-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------




    //Chain 4 is the Finger 5
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                               -------    minY/maxY      minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,0.0,  -8.0,25.0,   -90.0,10.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     2.0,     13.7 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-1.l",NO_ALTERNATE_NAME,BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -90.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,      0.0,     13.9 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-2.l",NO_ALTERNATE_NAME,CLOSEST_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
    //                                 -------   -------     minZ/maxZ
    addLimitsToNextPartOfChain(&data,  0.0,0.0,  0.0,0.0,    -45.0,0.0);
    //                                                         mAE X     mAE Y    mAE Z
    addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   0.0,     0.0,     4.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger5-3.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger5-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------




    #if DUALTHUMB
    //Thumb is complex so it has a minichain
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);

     expectBigValueChangesForNextPart(&data);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lthumbBase","__lthumb",BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);

     //                                                    minX/maxX   minY/maxY    minZ/maxZ
     //addLimitsToNextPartOfChain(&data,   0.0,35.0,  -60.0,0.0,   -60.0,0.0);
     //                                                         mAE X     mAE Y    mAE Z
     //addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   3.0,     2.6,     2.6 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lthumb",NO_ALTERNATE_NAME,CLOSEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger1-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
    #endif






    //Chain 5 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                  minX/maxX     minY/maxY   minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -35.0,0.0,    0.0,60.0,    0.0,60.0);
     //                                                        mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   13.0,     20.0,     20.0 );
     //addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   3.0,     2.6,     2.6 );
     ++correct; checksum+=addNewPartToChainProblem(&data,"lthumbBase","__lthumb",BASE_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
     //                                 minX/maxX     minY/maxY    minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -30.0,48.0,   0.0,85.0,   -85.0,85.0);
     //                                                         mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   15.3,     40.0,     40.0 );
     //addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   5.5,     14.8,     11.1 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lthumb",NO_ALTERNATE_NAME,CLOSEST_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
     //                                 minX/maxX    minY/maxY      0minZ/maxZ
     addLimitsToNextPartOfChain(&data, -40.0,45.0,   -70.0,35.0,   -35.0,0.0);
     //                                                        mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   6.7,     6.1,      2.8 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger1-2.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT); // should this be 0 ?
     //                                 minX/maxX    -------     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  0.0,50.0,    0.0,0.0,    -50.0,0.0);
     //                                                        mAE X     mAE Y    mAE Z
     addEstimatedMAEToPartOfChain_BeforeAddingNewPart(&data,   9.0,     0.0,      3.9 );
     ++correct;   checksum+=addNewPartToChainProblem(&data,"finger1-3.l",NO_ALTERNATE_NAME,MEDIAN_ENDPOINT_IMPORTANCE,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_finger1-3.l",NO_ALTERNATE_NAME,FURTHEST_ENDPOINT_IMPORTANCE,END_EFFECTOR);
      //----------------------------------------------------------
      if (failedPreparingChain(&data,correct,checksum)) { return 0; }
      //----------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts Fingers can be solved in parallel
     nextChain(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------

    problem->numberOfChains = data.chainID;
    problem->numberOfJobs = data.jobID;

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
    //----------------------------------------------------------
     float allTuningInOne[]     = {0.5,1.0,1.5,2.0,2.5};
     float MINIMAL_IMPORTANCE   = allTuningInOne[0];
     float LOW_IMPORTANCE       = allTuningInOne[1];
     float MEDIUM_IMPORTANCE    = allTuningInOne[2];
     float HIGH_IMPORTANCE      = allTuningInOne[3];
     float VERY_HIGH_IMPORTANCE = allTuningInOne[4];
    //----------------------------------------------------------

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
    unsigned int checksum=0;
    unsigned int correct=0;
    struct problemData data = {0};
    data.problem = problem;
    data.mc      = mc;
    //----------------------------------------------------------

     //First chain is the Hip and all of the rigid torso
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);

     expectBigValueChangesForNextPart(&data); //Big changes
     ++correct;   checksum+=addNewPartToChainProblemDetailed(
                                                             problem,mc,
                                                             //-----------------------------------------
                                                             "hip","Hips",HIGH_IMPORTANCE,OPTIMIZE_JOINT,
                                                             //-----------------------------------------
                                                             &data.jobID,&data.chainID,&data.partID,
                                                             //-----------------------------------------
                                                             1, //Force specific mIDStart/mIDEnd
                                                             0, //We have a position which since it comes from root joint should start at 0
                                                             2  //We have a position which since it comes from root joint should end at 2
                                                            );
     //expectSmallValueChangesForNextPart(&data); //Small changes for rotation? (not)
     ++correct;   checksum+=addNewPartToChainProblemDetailed(
                                                             problem,mc,
                                                             //-----------------------------------------
                                                             "hip","Hips",MINIMAL_IMPORTANCE,OPTIMIZE_JOINT,
                                                             //-----------------------------------------
                                                             &data.jobID,&data.chainID,&data.partID,
                                                             //-----------------------------------------
                                                             1, //Force specific mIDStart/mIDEnd
                                                             3, //We have a rotation which since it comes from root joint should start at 3
                                                             5  //We have a rotation which since it comes from root joint should end at 5
                                                            );
     //45.38
     ++correct;   checksum+=addNewPartToChainProblem(&data,"neck1","neck", MINIMAL_IMPORTANCE,END_EFFECTOR); //If neck1 is not available ( mnet1-mnet3 ) fallback to old neck
     ++correct;   checksum+=addNewPartToChainProblem(&data,"head",NO_ALTERNATE_NAME,  LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.l","eye.l",   MEDIUM_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.r","eye.r",   MEDIUM_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"ear.l","__temporalis02.l",MEDIUM_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"ear.r","__temporalis02.r",MEDIUM_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rshoulder","rShldr",      LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"relbow","rForeArm",       LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rhand",NO_ALTERNATE_NAME, MINIMAL_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lshoulder","lShldr",      LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lelbow","lForeArm",       LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lhand",NO_ALTERNATE_NAME, MINIMAL_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rhip","rThigh",           LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rknee","rShin",           LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rfoot",NO_ALTERNATE_NAME, MINIMAL_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lhip","lThigh",           LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lknee","lShin",           LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lfoot",NO_ALTERNATE_NAME, MINIMAL_IMPORTANCE,END_EFFECTOR);
    //-------------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //-------------------------------------------------------------
    nextChain(&data);
    //-------------------------------------------------------------
    //-------------------------------------------------------------


    #if USE_CHEST
     //Next chain is the Chest/Neck area ? There is no point
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //Unfortunately there is no Chest/Abdomen joint in the Body25 and trying to solve for one introduces wobblyness..
     ++correct;   checksum+=addNewPartToChainProblem(&data,"abdomen",NO_ALTERNATE_NAME,MINIMAL_IMPORTANCE,OPTIMIZE_JOINT);
     //                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,45.0,  -45.0,45.0,   -15.0,15.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"chest",NO_ALTERNATE_NAME,MINIMAL_IMPORTANCE,OPTIMIZE_JOINT);
     //                                  minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,45.0,  -45.0,45.0,   -15.0,15.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"neck1","neck", MINIMAL_IMPORTANCE,END_EFFECTOR); //If neck1 is not available ( mnet1-mnet3 ) fallback to old neck
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rshoulder","rShldr",     LOW_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lshoulder","lForeArm",   LOW_IMPORTANCE,END_EFFECTOR);
     //----------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //----------------------------------------------------------
     nextChainAndJob(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------
    #endif // USE_CHEST


     //Next chain is the Head
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                               minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,10.0,  -22.0,22.0,   -15.0,15.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"neck1","neck",  MINIMAL_IMPORTANCE  ,OPTIMIZE_JOINT); //If neck1 is not available ( mnet1-mnet3 ) fallback to old neck
     ++correct;   checksum+=addNewPartToChainProblem(&data,"head",NO_ALTERNATE_NAME,   HIGH_IMPORTANCE     ,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.l","eye.l",    VERY_HIGH_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_eye.r","eye.r",    VERY_HIGH_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"ear.l","__temporalis02.l", VERY_HIGH_IMPORTANCE,END_EFFECTOR);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"ear.r","__temporalis02.r", VERY_HIGH_IMPORTANCE,END_EFFECTOR);
     //This causes the head to tilt up!?
     //++correct;   checksum+=addNewPartToChainProblem(&data,"special04",NO_ALTERNATE_NAME, MEDIUM_IMPORTANCE,END_EFFECTOR); //"oris02"
     //-------------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //-------------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
     nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


     //Next chain is the R Shoulder
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //./getBVHColumnStats.sh generated/bvh_upperbody_all.csv 25 26 27
     //                                  minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -103.6,104.2,  -192.3,194.6,  -194.54,194.91);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rshoulder","rShldr",      MINIMAL_IMPORTANCE,  OPTIMIZE_JOINT);
     //                                  minX/maxX      minY/maxY       minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -68.5,8.37,    -110.0,164.0,   -47.34,35.64);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"relbow","rForeArm",       MEDIUM_IMPORTANCE,      OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rhand",NO_ALTERNATE_NAME, VERY_HIGH_IMPORTANCE,   END_EFFECTOR);
    //-------------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //-------------------------------------------------------------
    thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


     //Next chain  is the L Shoulder
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //./getBVHColumnStats.sh generated/bvh_upperbody_all.csv 34 35 36
     //                                  minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,  -103.6,104.2,  -192.3,194.6,  -194.54,194.91);
     ++correct; checksum+=addNewPartToChainProblem(&data,"lshoulder","lShldr",        MINIMAL_IMPORTANCE,   OPTIMIZE_JOINT);
     //                                  minX/maxX     minY/maxY     minZ/maxZ
     //Original values where .. addLimitsToNextPartOfChain(&data,  -68.5,9.5,    -163.7,15.7,   -12.36,47.64);
     addLimitsToNextPartOfChain(&data,  -68.5,9.5,    -164.0,110.0,   -35.64,47.64); //changed to mirror right hand
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lelbow","lForeArm",       MEDIUM_IMPORTANCE,       OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lhand",NO_ALTERNATE_NAME, VERY_HIGH_IMPORTANCE,    END_EFFECTOR);
    //-------------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //-------------------------------------------------------------
    thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


     //Next chain is the Right Foot
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, -135.0,45.0,  -40.0,50.0,   -10.0,80.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rhip","rThigh",                     MINIMAL_IMPORTANCE,  OPTIMIZE_JOINT);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,135.0,    0.0,0.0,    -10.0,10.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rknee","rShin",                     MEDIUM_IMPORTANCE,      OPTIMIZE_JOINT);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,28.0,    0.0,0.0,    -35.0,35.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rfoot",NO_ALTERNATE_NAME,           HIGH_IMPORTANCE,OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe1-2.r",NO_ALTERNATE_NAME,HIGH_IMPORTANCE,   END_EFFECTOR);  // Big Toe
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe5-3.r",NO_ALTERNATE_NAME,MEDIUM_IMPORTANCE,   END_EFFECTOR);  // Small Toe
     //-------------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //-------------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
     nextChainAndJob(&data);
     //----------------------------------------------------------
     //----------------------------------------------------------

     #if DUALFOOT
     //Next chain is the Right Sole
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,28.0,    0.0,0.0,    -35.0,35.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"rfoot",NO_ALTERNATE_NAME,             MEDIUM_IMPORTANCE,     OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe1-2.r",NO_ALTERNATE_NAME,  HIGH_IMPORTANCE,  END_EFFECTOR); // Big Toe
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe5-3.r",NO_ALTERNATE_NAME,  MEDIUM_IMPORTANCE,  END_EFFECTOR); // Small Toe
     //-------------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //-------------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
     nextChainAndJob(&data);
     //----------------------------------------------------------
    #endif

     //Next chain  is the Left Foot
     //----------------------------------------------------------
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                minX/maxX     minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, -135.0,45.0,  -50.0,40.0,   -80.0,10.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lhip","lThigh",                            MINIMAL_IMPORTANCE,    OPTIMIZE_JOINT);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data, 0.0,135.0,    0.0,0.0,    -10.0,10.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lknee","lShin",                            MEDIUM_IMPORTANCE,        OPTIMIZE_JOINT);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,28.0,    0.0,0.0,    -35.0,35.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lfoot",NO_ALTERNATE_NAME,                  HIGH_IMPORTANCE,        OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe1-2.l",NO_ALTERNATE_NAME,       HIGH_IMPORTANCE,        END_EFFECTOR); // Big Toe
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe5-3.l",NO_ALTERNATE_NAME,       MEDIUM_IMPORTANCE,      END_EFFECTOR); // Small Toe
    //-------------------------------------------------------------
    if (failedPreparingChain(&data,correct,checksum)) { return 0; }
    //-------------------------------------------------------------
    thisChainCanBeRunInParallel(&data);//This has to be done after adding parts / Limbs can be solved in parallel
    nextChainAndJob(&data);
    //----------------------------------------------------------
    //----------------------------------------------------------


    #if DUALFOOT
     //Next chain  is the Left Sole
     //----------------------------------------------------------
     checksum=0; correct=0; startAddingNewPartsToChain(&data);
     //                                minX/maxX    minY/maxY     minZ/maxZ
     addLimitsToNextPartOfChain(&data,-10.0,28.0,    0.0,0.0,    -35.0,35.0);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"lfoot",NO_ALTERNATE_NAME,                MEDIUM_IMPORTANCE,          OPTIMIZE_JOINT);
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe1-2.l",NO_ALTERNATE_NAME,     HIGH_IMPORTANCE,       END_EFFECTOR); // Big Toe
     ++correct;   checksum+=addNewPartToChainProblem(&data,"endsite_toe5-3.l",NO_ALTERNATE_NAME,     MEDIUM_IMPORTANCE,       END_EFFECTOR); // Small Toe
     //-------------------------------------------------------------
     if (failedPreparingChain(&data,correct,checksum)) { return 0; }
     //-------------------------------------------------------------
     thisChainCanBeRunInParallel(&data); //This has to be done after adding parts / Limbs can be solved in parallel
     nextChainAndJob(&data);
     //----------------------------------------------------------
    #endif

    problem->numberOfChains = data.chainID;
    problem->numberOfJobs   = data.jobID;

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
    char command[1025]={0};
    if (dumpScreenshots)
    {
        snprintf(command,1024,"convert initial.svg initial_%u_%u.png&",fIDSource,fIDTarget);
        int i=system(command); //"convert initial.svg initial.png&"
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }
        //-----------------------------------------------------------------------------------
        snprintf(command,1024,"convert target.svg target_%u_%u.png&",fIDSource,fIDTarget);
        i=system(command); //"convert target.svg target.png&"
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }
        //-----------------------------------------------------------------------------------
        snprintf(command,1024,"convert solution.svg solution_%u_%u.png&",fIDSource,fIDTarget);
        i=system(command); //"convert solution.svg solution.png&"
        if (i!=0)
        {
            fprintf(stderr,"Error converting image..\n");
        }
        //-----------------------------------------------------------------------------------

        snprintf(command,1024,"report_%u_%u.html",fIDSource,fIDTarget);
        FILE * html=fopen(command,"w"); //"report.html"
        if (html!=0)
        {
            fprintf(html,"<html><body><br>\n");
            //------------------------------------------------------------
            fprintf(html,"<table><tr>\n");
            fprintf(html,"<td><img src=\"initial_%u_%u.png\" width=400></td>\n",fIDSource,fIDTarget);
            fprintf(html,"<td><img src=\"target_%u_%u.png\" width=400></td>\n",fIDSource,fIDTarget);
            fprintf(html,"<td><img src=\"solution_%u_%u.png\" width=400></td>\n",fIDSource,fIDTarget);
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

float bvhTestIK(
                struct BVH_MotionCapture * mc,
                float lr,
                float spring,
                unsigned int iterations,
                unsigned int epochs,
                float langevin,
                float learningRateDecayRate,
                float momentum,
                unsigned int fIDPrevious,
                unsigned int fIDSource,
                unsigned int fIDTarget,
                unsigned int multiThreaded,
                char verboseAndDumpFiles
               )
{
    //int result=0;

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
                    //--------------------------------------------------
                    ikConfig.learningRate                  = lr;
                    ikConfig.iterations                    = iterations;
                    ikConfig.epochs                        = epochs;
                    ikConfig.spring                        = spring;
                    ikConfig.gradientExplosionThreshold    = 50;
                    ikConfig.iterationEarlyStopping        = 1;  //<-
                    ikConfig.iterationMinimumLossDelta     = 10; //<- losses seem to be numbers 2000 -> 300 so 10 is a good limit
                    ikConfig.maximumAcceptableStartingLoss = 0.0; // Dont use this
                    ikConfig.dumpScreenshots               = verboseAndDumpFiles;
                    ikConfig.verbose                       = verboseAndDumpFiles;
                    ikConfig.tryMaintainingLocalOptima     = 1; //Less Jittery but can be stuck at local optima
                    ikConfig.dontUseSolutionHistory        = 0;
                    ikConfig.useLangevinDynamics           = langevin;
                    ikConfig.learningRateDecayRate         = learningRateDecayRate;
                    ikConfig.hcdMomentum                   = momentum;
                    ikConfig.ikVersion = IK_VERSION;
                    //--------------------------------------------------

                    struct ikProblem * problem= (struct ikProblem * ) malloc(sizeof(struct ikProblem));
                    if (problem!=0)
                                     { memset(problem,0,sizeof(struct ikProblem)); } else
                                     { fprintf(stderr,"Failed to allocate memory for our IK problem..\n");  return 0; }

                    if (
                        !prepareDefaultBodyProblem(
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
                        //result=1;
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
    return finalMAEInMM*10;
}
