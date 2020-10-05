#include "hardcodedProblems_inverseKinematics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//This causes a double free.. :S ( double free or corruption (!prev) )
#define DUALFOOT 1

int addNewPartToChainProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    //-----------------------------------------
    char * partName,
    char * alternatePartName,
    float importance,
    int isEndEffector,
    unsigned int * groupID,
    unsigned int * jobID,
    unsigned int * chainID,
    unsigned int * partID
    )
{ 
    if (*chainID >= MAXIMUM_CHAINS)
    {
      fprintf(stderr,RED "Reached limit of maximum chains.. (%u) \n" NORMAL,MAXIMUM_CHAINS);
      return 0;  
    }
    
    if (*partID >= MAXIMUM_PARTS_OF_CHAIN)
    {
      fprintf(stderr,RED "Reached limit of maximum parts of the chain.. (%u) \n" NORMAL,MAXIMUM_PARTS_OF_CHAIN);
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
        problem->chain[*chainID].part[*partID].evaluated=0; //Not evaluated yet
        problem->chain[*chainID].part[*partID].jID=thisJID; 
        problem->chain[*chainID].part[*partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
        problem->chain[*chainID].part[*partID].mIDEnd=problem->chain[*chainID].part[*partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
        problem->chain[*chainID].part[*partID].jointImportance=importance;
        problem->chain[*chainID].part[*partID].endEffector=isEndEffector;
        
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
        exit(0);
        return 0;
    }
}



int prepareDefaultFaceProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
)
{ 
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

    snprintf(problem->problemDescription,64,"Face");


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
    BVHJointID thisJID=0;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "orbicularis3.r",0,  // Joint 
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_orbicularis3.r",0,  // Joint 
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "orbicularis4.r",0,  // Joint 
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                                  
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_orbicularis4.r",0,  // Joint 
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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

     return 1;
}






int prepareDefaultRightHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
)
{
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
    
    snprintf(problem->problemDescription,64,"Right Hand");


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
    BVHJointID thisJID=0;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rshoulder","rShldr",  // Joint 
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
      
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "relbow","rForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
   
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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






     //CHAIN 0 ----------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     //---------------------------------------------------------- 
     checksum=0;
     correct=0;
     partID=0;
     
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rhand",0, // Joint
                               2.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
                             
    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
         
    ++chainID;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-1.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );

    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
                             
    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel  
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------




    //Chain 5 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------
     checksum=0;
     correct=0;
     partID=0; // Reset counter..
     
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rthumb",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger1-2.r",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger1-3.r",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
                             
    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }                     
                             
    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel  
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------

    problem->numberOfChains = chainID;
    problem->numberOfGroups = groupID;
    problem->numberOfJobs = jobID;
    
  return 1;
}











int prepareDefaultLeftHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
)
{
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

    snprintf(problem->problemDescription,64,"Left Hand");


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
    BVHJointID thisJID=0;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lshoulder","lShldr",  // Joint 
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
      
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lelbow","lForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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




     //CHAIN 0 ----------------------------
     //----------------------------------------------------------
     //----------------------------------------------------------
     //---------------------------------------------------------- 
     checksum=0;
     correct=0;
     partID=0;
     
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lhand",0, // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-1.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-1.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-1.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-1.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lthumb",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
                             
    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
         
    ++chainID;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-1.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-2.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger2-3.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-1.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-2.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger3-3.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-1.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-2.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger4-3.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-1.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-2.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger5-3.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );

    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }
                             
    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel  
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------


    //Chain 5 is the Finger 1 ( Thumb )
    //----------------------------------------------------------
    //----------------------------------------------------------
    //----------------------------------------------------------      
     checksum=0;
     correct=0;
     partID=0; // Reset counter..
     
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lthumb",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger1-2.l",0, // Joint
                              1.0,     //Importance
                              0,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "finger1-3.l",0, // Joint
                              1.0,     //Importance
                              1,       //IsEndEffector
                              &groupID,&jobID,&chainID,&partID
                             );
                             
    if (correct!=checksum) 
         { fprintf(stderr,"Failed at Chain %u (%u/%u)\n",chainID,checksum,correct); return 0; }                       
                             
    problem->chain[chainID].parallel=1; //This has to be done after adding parts Fingers can be solved in parallel  
     ++chainID;
    //----------------------------------------------------------
    //----------------------------------------------------------

    problem->numberOfChains = chainID;
    problem->numberOfGroups = groupID;
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

    snprintf(problem->problemDescription,64,"Body");


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
    BVHJointID thisJID=0;
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "hip",0,    // Joint
                               2.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     problem->chain[chainID].part[partID-1].bigChanges=1; //Big changes
     problem->chain[chainID].part[partID-1].mIDStart=0; //First Position
     problem->chain[chainID].part[partID-1].mIDEnd=2; //First Position
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "hip",0,    // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
     problem->chain[chainID].part[partID-1].mIDStart=3; //First Position
     problem->chain[chainID].part[partID-1].mIDEnd=5; //First Position        

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "neck",0,    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );                   

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "head",0,    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );            
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rshoulder","rShldr", // Joint 
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );                   

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lshoulder","lShldr",      // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );                   
 
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rhip","rThigh", // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );             

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lhip","lThigh", // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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
    
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "chest",0,// Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
   
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "neck",0, // Joint
                               0.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rshoulder","rShldr", // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lshoulder","lForeArm",    // Joint
                               1.0,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "neck",0,  // Joint 
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
      
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "head",0,  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "eye.l",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             

     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "eye.r",0,// Joint
                               2.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rshoulder","rShldr",  // Joint 
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
      
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "relbow","rForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rhand",0,// Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lshoulder","lShldr",   // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lelbow","lForeArm",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lhand",0,  // Joint
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rhip","rThigh", // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rknee","rShin", // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "rfoot",0,  // Joint
                               1.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                                                         
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_toe1-2.r",0,  // Big Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
               
     #if DUALFOOT              
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_toe5-3.r",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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

 
 



 
     //Next chain  is the Left Foot
     //----------------------------------------------------------
     //----------------------------------------------------------
     //---------------------------------------------------------- 
     checksum=0;
     correct=0; 
     partID=0;
    
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lhip","lThigh",  // Joint
                               0.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lknee","lShin",  // Joint
                               1.0,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                             
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "lfoot",0,// Joint
                               1.5,     //Importance
                               0,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
                                                         
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_toe1-2.l",0, // Big Toe 
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
                             );
   
     #if DUALFOOT
     ++correct;
     checksum+=addNewPartToChainProblem(
                              problem,mc,renderer,previousSolution,solution,bvhTargetTransform,
                              //-----------------------------------------
                              "endsite_toe5-3.l",0,  // Small Toe
                               1.5,     //Importance
                               1,       //IsEndEffector
                              //-----------------------------------------
                              &groupID,&jobID,&chainID,&partID
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



    ++groupID;

    problem->numberOfChains = chainID;
    problem->numberOfGroups = groupID;
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
        freeMotionBuffer(&previousSolution);
        freeMotionBuffer(&solution);
        freeMotionBuffer(&initialSolution);
        freeMotionBuffer(&groundTruth);
    }

    return result;
}




