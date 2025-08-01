/*
 * bvh_inverseKinematics.c
 *
 * This file contains an implementation of my inverse kinematics algorithm
 *
 * Ammar Qammaz, May 2020
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <assert.h> //assert support for debugging
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include <time.h>

#include "hardcodedProblems_inverseKinematics.h"
#include "bvh_inverseKinematics.h"
#include "levmar.h"

#include "../edit/bvh_remapangles.h"
#include "../export/bvh_to_svg.h"
#include "../edit/bvh_cut_paste.h"

#include "../../../../../../tools/PThreadWorkerPool/pthreadWorkerPool.h"

// --------------------------------------------
#include <errno.h>
// --------------------------------------------

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


unsigned long tickBaseIK = 0;

void clear_line()
{
    fputs("\033[A\033[2K\033[A\033[2K",stdout);
    rewind(stdout);
    int i=ftruncate(1,0);
    if (i!=0)
    {
        /*fprintf(stderr,"Error with ftruncate\n");*/
    }
}

char fileExistsIK(const char * filename)
{
    FILE *fp = fopen(filename,"r");
    if(fp)
        { //exists
          fclose(fp);
          return 1;
        }
    return 0;
}

unsigned long GetTickCountMicrosecondsIK()
{
    struct timespec ts;
    if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
    {
        return 0;
    }

    if (tickBaseIK==0)
    {
        tickBaseIK = ts.tv_sec*1000000 + ts.tv_nsec/1000;
        return 0;
    }

    return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseIK;
}

unsigned long GetTickCountMillisecondsIK()
{
    return (unsigned long) GetTickCountMicrosecondsIK()/1000;
}

float getSquared3DPointDistance(float aX,float aY,float aZ,float bX,float bY,float bZ)
{
    float diffX = (float) aX-bX;
    float diffY = (float) aY-bY;
    float diffZ = (float) aZ-bZ;
    //We calculate the distance here..!
    return (diffX*diffX) + (diffY*diffY) + (diffZ*diffZ);
}

float get3DPointDistance(float aX,float aY,float aZ,float bX,float bY,float bZ)
{
    return sqrt(getSquared3DPointDistance(aX,aY,aZ,bX,bY,bZ));
}

float getSquared2DPointDistance(float aX,float aY,float bX,float bY)
{
    float diffX = (float) (aX-bX);
    float diffY = (float) (aY-bY);
    //We calculate the distance here..!
    return (diffX*diffX) + (diffY*diffY);
}

float get2DPointDistance(float aX,float aY,float bX,float bY)
{
    return sqrt(getSquared2DPointDistance(aX,aY,bX,bY));
}

float meanBVH2DDistance(
                        struct BVH_MotionCapture * mc,
                        struct simpleRenderer *renderer,
                        int useAllJoints,
                        BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
                        struct BVH_Transform * bvhSourceTransform,
                        struct BVH_Transform * bvhTargetTransform,
                        unsigned int verbose
                       )
{
    if (verbose)
    {
        fprintf(stderr,"\nmeanBVH2DDistance\n");
    }
    if (bvh_projectTo2D(mc,bvhSourceTransform,renderer,0,0))
    {
        //-----------------
        float sumOf2DDistances=0.0;
        unsigned int numberOfSamples=0;
        for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
        {
            int isSelected = 1;

            if (mc->selectedJoints!=0)
            {
              //Selected joints table is declared so we can access it
              isSelected=mc->selectedJoints[jID];
            }

            if ( (useAllJoints) || ( (isSelected) && (mc->jointHierarchy[jID].parentJoint == onlyConsiderChildrenOfThisJoint) ) )
            {
                ///Warning: When you change this please change calculateChainLoss as well!
                float sX=bvhSourceTransform->joint[jID].pos2D[0];
                float sY=bvhSourceTransform->joint[jID].pos2D[1];
                float tX=bvhTargetTransform->joint[jID].pos2D[0];
                float tY=bvhTargetTransform->joint[jID].pos2D[1];
                float notZeroIfAllPointsExist = sX*sY*tX*tY;

                //if ( ( (sX!=0.0) || (sY!=0.0) ) && ( (tX!=0.0) || (tY!=0.0) ) )
                if (notZeroIfAllPointsExist!=0.0)
                {
                    float this2DDistance=get2DPointDistance(sX,sY,tX,tY);
                    if (verbose)
                    {
                        fprintf(stderr,"src(%0.1f,%0.1f)->tar(%0.1f,%0.1f) : 2D ",sX,sY,tX,tY);
                        if (mc->jointHierarchy[jID].jointName!=0) { fprintf(stderr,"%s ",mc->jointHierarchy[jID].jointName);}
                        fprintf(stderr,"distance = %0.1f\n",this2DDistance);
                    }
                    numberOfSamples+=1;
                    sumOf2DDistances+=this2DDistance;
                }
            }
        }
        if (verbose)
        {
            fprintf(stderr,"\n");
        }

        if (numberOfSamples>0)
        {
            return (float)  sumOf2DDistances/numberOfSamples;
        }
    } else //-----------------
    {
     fprintf(stderr,"meanBVH2DDistance failed bvh_projectTo2D call\n");
    }


    return 0.0;
}

float meanBVH3DDistance(
                        struct BVH_MotionCapture * mc,
                        struct simpleRenderer *renderer,
                        int useAllJoints,
                        BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
                        float * sourceMotionBuffer,
                        struct BVH_Transform * bvhSourceTransform,
                        float * targetMotionBuffer,
                        struct BVH_Transform * bvhTargetTransform
                       )
{
    if (targetMotionBuffer==0)
          { return NAN; }

    if (
        (
         performPointProjectionsForMotionBuffer(mc,bvhSourceTransform,sourceMotionBuffer,renderer,0,0)
        ) &&
        (
         performPointProjectionsForMotionBuffer(mc,bvhTargetTransform,targetMotionBuffer,renderer,0,0)
        )
       )
    {
        //-----------------
        float sumOf3DDistances=0.0;
        unsigned int numberOfSamples=0;
        for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
        {
            int isSelected = 1;

            if (mc->selectedJoints!=0)
            {
              //Selected joints table is declared so we can access it
              isSelected=mc->selectedJoints[jID];
            }

            if ( (isSelected) && ( (useAllJoints) || (mc->jointHierarchy[jID].parentJoint == onlyConsiderChildrenOfThisJoint) ) )
            {
                float tX=bvhTargetTransform->joint[jID].pos3D[0];
                float tY=bvhTargetTransform->joint[jID].pos3D[1];
                float tZ=bvhTargetTransform->joint[jID].pos3D[2];
                float notZeroIfAllPointsExist = tX*tY*tZ;

                //if ( (tX!=0.0) || (tY!=0.0) || (tZ!=0.0) )
                if (notZeroIfAllPointsExist!=0.0)
                {
                    float this3DDistance=get3DPointDistance(
                                             (float) bvhSourceTransform->joint[jID].pos3D[0],
                                             (float) bvhSourceTransform->joint[jID].pos3D[1],
                                             (float) bvhSourceTransform->joint[jID].pos3D[2],
                                             (float) tX,
                                             (float) tY,
                                             (float) tZ
                                         );

                    fprintf(stderr,"src(%0.1f,%0.1f,%0.1f)->tar(%0.1f,%0.1f,%0.1f) : ",(float) bvhSourceTransform->joint[jID].pos3D[0],
                            (float) bvhSourceTransform->joint[jID].pos3D[1],
                            (float) bvhSourceTransform->joint[jID].pos3D[2],
                            (float) tX,
                            (float) tY,
                            (float) tZ);
                    fprintf(stderr," %s distance^2 = %0.1f\n",mc->jointHierarchy[jID].jointName,this3DDistance);

                    numberOfSamples+=1;
                    sumOf3DDistances+=this3DDistance;
                }
            }
        }

        if (numberOfSamples>0)
        {
            return (float)  sumOf3DDistances/numberOfSamples;
        }
    } //-----------------

    return 0.0;
}

int updateProblemSolutionToAllChains(struct ikProblem * problem,struct MotionBuffer * updatedSolution)
{
    if (problem==0)                   { fprintf(stderr,"updateProblemSolutionToAllChains: No Problem\n");          return 0; }
    if (updatedSolution==0)           { fprintf(stderr,"updateProblemSolutionToAllChains: No updated solution\n"); return 0; }
    if (problem->currentSolution==0)  { fprintf(stderr,"updateProblemSolutionToAllChains: No currentSolution\n");  return 0; }
    if (problem->initialSolution==0)  { fprintf(stderr,"updateProblemSolutionToAllChains: No initialSolution\n");  return 0; }

    //Actual copy ------------------------------------------------------------------
    if (!copyMotionBuffer(problem->currentSolution,updatedSolution) )  {  fprintf(stderr,"updateProblemSolutionToAllChains: Failed updating currentSolution\n");  return 0; }
    if (!copyMotionBuffer(problem->initialSolution,updatedSolution) )  {  fprintf(stderr,"updateProblemSolutionToAllChains: Failed updating initialSolution\n");  return 0; }

    if (problem->numberOfChains >= MAXIMUM_CHAINS)
    {
      fprintf(stderr,"updateProblemSolutionToAllChains: Too many chains.. %u/%d \n",problem->numberOfChains,MAXIMUM_CHAINS);
      return 0;
    }

    for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
    {
        if (!copyMotionBuffer(problem->chain[chainID].currentSolution,updatedSolution))
        {
            fprintf(stderr,"updateProblemSolutionToAllChains: Failed updating currentSolution for chain %u\n",chainID);
            return 0;
        }
    }
    return 1;
}

int cleanProblem(struct ikProblem * problem)
{
    if (problem==0) { fprintf(stderr,"Cannot clean null problem\n"); return 0; }

    freeMotionBuffer(&problem->previousSolution);
    freeMotionBuffer(&problem->initialSolution);
    freeMotionBuffer(&problem->currentSolution);

    if (problem->numberOfChains >= MAXIMUM_CHAINS)
    {
        fprintf(stderr,"Cannot clean problem with overflowing number of chains %u/%d \n",problem->numberOfChains,MAXIMUM_CHAINS);
        return 0;
    }

    for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
    {
        freeMotionBuffer(&problem->chain[chainID].currentSolution);
        //Terminate all threads..
        problem->chain[chainID].terminate=1;
        problem->chain[chainID].threadIsSpawned=0;
    }

   if (problem->threadPool.initialized)
   {
       if (!threadpoolDestroy(&problem->threadPool))
       {
           fprintf(stderr,"Failed deleting IK thread pool\n");
       }
   }

    return 1;
}


int viewProblem(struct ikProblem * problem)
{
    fprintf(stderr,"\n\n\n\n");
    fprintf(stderr,"The IK problem " GREEN "\"%s\"" NORMAL " we want to solve \n",problem->problemDescription); //problem->numberOfGroups
    fprintf(stderr,"is ultimately divided into %u kinematic chains\n",problem->numberOfChains);

    for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
    {
        fprintf(stderr,GREEN "Chain %u has %u parts : " NORMAL,chainID,problem->chain[chainID].numberOfParts);
        for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
        {
            unsigned int jID=problem->chain[chainID].part[partID].jID;

            if (problem->chain[chainID].part[partID].endEffector)
            {
                fprintf(stderr,"jID(%s/%u)->EndEffector ",problem->mc->jointHierarchy[jID].jointName,jID);
            }
            else
            {
                fprintf(stderr,"jID(%s/%u)->mID(%u to %u) ",
                              problem->mc->jointHierarchy[jID].jointName,
                              jID,
                              problem->chain[chainID].part[partID].mIDStart,
                              problem->chain[chainID].part[partID].mIDEnd
                            );
            }
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"End of problem " GREEN "\"%s\"" NORMAL " \n",problem->problemDescription); //problem->numberOfGroups

    return 1;
}




float calculateChainLoss(
                          struct ikProblem * problem,
                          unsigned int chainID,
                          unsigned int partIDStart,
                          unsigned int penalizeSymmetryIn,
                          unsigned int economicTransformCalculation
                        )
{
    unsigned int numberOfSamples=0;
    float loss=0;
    if (chainID<problem->numberOfChains)
    {
      if (partIDStart < problem->chain[chainID].numberOfParts)
      {
        unsigned int transformIsLoaded=0;

        if (economicTransformCalculation)
        {
            //IMPORTANT: Please note that this call relies on the internal accumulation of jIDs that happens on a regular
            //bvh_loadTransformForMotionBuffer call however calling it once before is not guaranteed..
            //So be very very careful..
            transformIsLoaded = bvh_loadTransformForMotionBufferFollowingAListOfJointIDs
                                                                (
                                                                  problem->mc,
                                                                  problem->chain[chainID].currentSolution->motion,
                                                                  &problem->chain[chainID].current2DProjectionTransform,
                                                                  penalizeSymmetryIn,//Dont populate extra structures we dont need them they just take time
                                                                  problem->chain[chainID].current2DProjectionTransform.listOfJointIDsToTransform,
                                                                  problem->chain[chainID].current2DProjectionTransform.lengthOfListOfJointIDsToTransform
                                                                );
        }  else
        {
            transformIsLoaded = bvh_loadTransformForMotionBuffer(
                                                                  problem->mc,
                                                                  problem->chain[chainID].currentSolution->motion,
                                                                  &problem->chain[chainID].current2DProjectionTransform,
                                                                  penalizeSymmetryIn//Dont populate extra structures we dont need them they just take time
                                                                );
        }



        if (transformIsLoaded)
        {
           //Only projecting parts is a performance measure..
           #define ONLY_PROJECT_PARTS 1

           #if ONLY_PROJECT_PARTS
           //To perform better we can skip projections that are not part of our kinematic chain..
           //We thus perform fewer matrix multiplications
           unsigned int failedProjections=0;
           for (unsigned int partID=partIDStart; partID<problem->chain[chainID].numberOfParts; partID++)
                {
                  unsigned int jID=problem->chain[chainID].part[partID].jID;
                  failedProjections += ( bvh_projectJIDTo2D(problem->mc,&problem->chain[chainID].current2DProjectionTransform,problem->renderer,jID,0,0) == 0 );
                }

           if (failedProjections==0)
            {
           #else
           //We project all Joints to their 2D locations..
            if  (bvh_projectTo2D(problem->mc,&problem->chain[chainID].current2DProjectionTransform,problem->renderer,0,0))
             {
           #endif
                for (unsigned int partID=partIDStart; partID<problem->chain[chainID].numberOfParts; partID++)
                {
                   //There should never be a joint with Importance 0.0, If the joint importance is zero it is bad problem design.. there shouldn't be millions of checks for this
                   //if (problem->chain[chainID].part[partID].jointImportance!=0.0)
                   {
                        BVHJointID jID=problem->chain[chainID].part[partID].jID;

                        ///Warning: When you change this please change meanBVH2DDistance as well!
                        float sX=(float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[0];
                        float sY=(float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[1];
                        float tX=(float) problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[0];
                        float tY=(float) problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[1];
                        float notZeroIfAllPointsExist = sX*sY*tX*tY;

                        //Only use source/target joints  that exist and are not occluded..
                        //if ( ((sX!=0.0) || (sY!=0.0)) && ((tX!=0.0) || (tY!=0.0)) )
                        if (notZeroIfAllPointsExist!=0.0)
                        {
                            loss+= getSquared2DPointDistance(sX,sY,tX,tY) * problem->chain[chainID].part[partID].jointImportance;
                            ++numberOfSamples;

                            //Add weight for backwards joints..
                            //This is a heuristic *patch* on the loss to prevent hands from bending backwards!
                            if (penalizeSymmetryIn)
                            { //PENALIZE_SYMMETRY_HEURISTIC
                              if (strstr(problem->mc->jointHierarchy[jID].jointName,"Hand")!=0 )
                              {//Crude and slow debug ..
                               float symmetriesLoss=bvh_DistanceOfJointFromTorsoPlane(
                                                                                      problem->mc,
                                                                                      &problem->chain[chainID].current2DProjectionTransform,
                                                                                      jID
                                                                                     );
                               symmetriesLoss += 15.0;

                               //Only negative contribution..
                               if ( (symmetriesLoss>10.0) || (problem->chain[chainID].current2DProjectionTransform.joint[jID].isOccluded) )
                                  {
                                      float gain = 225.0;
                                      symmetriesLoss = 1.0 * fabs(symmetriesLoss) * gain; // <- Easy flip
                                      //fprintf(stderr,"Symmetry Loss %0.2f / %0.2f @ %s \n",symmetriesLoss,loss,problem->mc->jointHierarchy[jID].jointName);
                                      loss+=symmetriesLoss;
                                  }
                              }
                            }
                        }
                   } //We might want to ignore the error of the particular joint, useful when observation is misaligned to hypothesis..
                } //We add ever part of this chain
            } else  // We successfully projected the BVH file to 2D points..
           { fprintf(stderr,RED "Could not calculate transform projections to 2D for chain %u \n"NORMAL,chainID); }
        } else //Have a valid 2D transform
       { fprintf(stderr,RED "Could not calculate transform for chain %u is invalid\n"NORMAL,chainID); }

     } else
     { fprintf(stderr,RED "Chain %u has too few parts ( %u ) \n" NORMAL,chainID,problem->chain[chainID].numberOfParts); }
    } else //Have a valid chain
    { fprintf(stderr,RED "Chain %u is invalid\n"NORMAL,chainID); }
    //I have left 0/0 on purpose to cause NaNs when projection errors occur
    //----------------------------------------------------------------------------------------------------------
    if (numberOfSamples!=0) { return (float) loss/numberOfSamples; }  else
                            { return NAN; }
    //----------------------------------------------------------------------------------------------------------
}



int examineSolutionAndKeepIfItIsBetterSingleTry(
                                                struct ikProblem * problem,
                                                unsigned int iterationID,
                                                unsigned int chainID,
                                                unsigned int partID,
                                                unsigned int * mIDS,
                                                float * originalValues,
                                                float * bestValues,
                                                float * bestLoss,
                                                float spring,
                                                //-------------------------
                                                float * solutionToTest,
                                                //-------------------------
                                                int penalizeSymmetryIn
                                               )
{
      float previousValues[3]={
                                problem->chain[chainID].currentSolution->motion[mIDS[0]],
                                problem->chain[chainID].currentSolution->motion[mIDS[1]],
                                problem->chain[chainID].currentSolution->motion[mIDS[2]]
                              };
        //Calculate loss of try
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        problem->chain[chainID].currentSolution->motion[mIDS[0]] = solutionToTest[0];
        problem->chain[chainID].currentSolution->motion[mIDS[1]] = solutionToTest[1];
        problem->chain[chainID].currentSolution->motion[mIDS[2]] = solutionToTest[2];
        float currentLoss =calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/) ;//+ spring * distanceFromInitial * distanceFromInitial;
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        if (currentLoss<*bestLoss)
        {
            *bestLoss = currentLoss;
            bestValues[0] = solutionToTest[0];
            bestValues[1] = solutionToTest[1];
            bestValues[2] = solutionToTest[2];
            return 1;
        } else
        { //Roll Back..!
          problem->chain[chainID].currentSolution->motion[mIDS[0]] = previousValues[0];
          problem->chain[chainID].currentSolution->motion[mIDS[1]] = previousValues[1];
          problem->chain[chainID].currentSolution->motion[mIDS[2]] = previousValues[2];
        }
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        return 0;
}

float randomNoise(float noiseMagnitude)
{
  float x = ((float)rand()/(float)(RAND_MAX)) * noiseMagnitude;
  return x-(float) (noiseMagnitude/2);
}

int examineSolutionAndKeepIfItIsBetter(
                                       struct ikProblem * problem,
                                       unsigned int iterationID,
                                       unsigned int chainID,
                                       unsigned int partID,
                                       unsigned int * mIDS,
                                       float * originalValues,
                                       float * bestValues,
                                       float * bestLoss,
                                       float spring,
                                       //-------------------------
                                       float * solutionToTest,
                                       //-------------------------
                                       int penalizeSymmetryIn
                                      )
{
        int accepted = 0;
        //---------------------------------------------------
        float previousValues[3]={
                                problem->chain[chainID].currentSolution->motion[mIDS[0]],
                                problem->chain[chainID].currentSolution->motion[mIDS[1]],
                                problem->chain[chainID].currentSolution->motion[mIDS[2]]
                              };
        //-------------------  -------------------  -------------------  -------------------
        // Calculate loss of try
        //-------------------  -------------------  -------------------  -------------------
        problem->chain[chainID].currentSolution->motion[mIDS[0]] = solutionToTest[0];
        float currentLoss = calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/) ;//+ spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss<*bestLoss)
                { *bestLoss = currentLoss; bestValues[0] = solutionToTest[0]; accepted+=1;       } else //Roll Back..!
                {  problem->chain[chainID].currentSolution->motion[mIDS[0]] = previousValues[0]; }
        //------------------------------------------------------------------------------
        problem->chain[chainID].currentSolution->motion[mIDS[1]] = solutionToTest[1];
        currentLoss = calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/) ;//+ spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss<*bestLoss)
                { *bestLoss = currentLoss; bestValues[1] = solutionToTest[1]; accepted+=1;       } else //Roll Back..!
                {  problem->chain[chainID].currentSolution->motion[mIDS[1]] = previousValues[1]; }
        //------------------------------------------------------------------------------
        problem->chain[chainID].currentSolution->motion[mIDS[2]] = solutionToTest[2];
        currentLoss = calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/) ;//+ spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss<*bestLoss)
                { *bestLoss = currentLoss; bestValues[2] = solutionToTest[2]; accepted+=1;       } else //Roll Back..!
                {  problem->chain[chainID].currentSolution->motion[mIDS[2]] = previousValues[2]; }
        //-------------------  -------------------  -------------------  -------------------
        return (accepted!=0);
}

void ensureValuesInLimits(float vals[3],float mins[3],float maxes[3])
{
 //-------------------------------------------------------------------------------------
 if (vals[0]<mins[0])  { vals[0]=mins[0];  } else
 if (vals[0]>maxes[0]) { vals[0]=maxes[0]; }
 //-------------------------------------------------------------------------------------
 if (vals[1]<mins[1])  { vals[1]=mins[1];  } else
 if (vals[1]>maxes[1]) { vals[1]=maxes[1]; }
 //-------------------------------------------------------------------------------------
 if (vals[2]<mins[2])  { vals[2]=mins[2];  } else
 if (vals[2]>maxes[2]) { vals[2]=maxes[2]; }
 //-------------------------------------------------------------------------------------
}

void updateLimitsBasedOnMAE(
                            struct ikProblem * problem,
                            unsigned int chainID,
                            unsigned int partID,
                            float originalValues[3],
                            float minimumLimitValues[3],
                            float maximumLimitValues[3]
                           )
{
 char limitsEngaged = problem->chain[chainID].part[partID].limits;

 //Update Minima ------------------------------------------------------------
 float newMinA = originalValues[0]-problem->chain[chainID].part[partID].mAE[0];
 float newMinB = originalValues[1]-problem->chain[chainID].part[partID].mAE[1];
 float newMinC = originalValues[2]-problem->chain[chainID].part[partID].mAE[2];

 //Update Maxima ------------------------------------------------------------
 float newMaxA = originalValues[0]+problem->chain[chainID].part[partID].mAE[0];
 float newMaxB = originalValues[1]+problem->chain[chainID].part[partID].mAE[1];
 float newMaxC = originalValues[2]+problem->chain[chainID].part[partID].mAE[2];

 //This unrolling does only one jump instruction and is thus faster
 if (limitsEngaged) {
                      if (newMinA<minimumLimitValues[0])  { newMinA=minimumLimitValues[0];  } else
                      if (newMaxA>maximumLimitValues[0])  { newMaxA=maximumLimitValues[0];  }
                      //---------------------------------------------------------------------------
                      if (newMinB<minimumLimitValues[1])  { newMinB=minimumLimitValues[1];  } else
                      if (newMaxB>maximumLimitValues[1])  { newMaxB=maximumLimitValues[1];  }
                      //---------------------------------------------------------------------------
                      if (newMinC<minimumLimitValues[2])  { newMinC=minimumLimitValues[2];  } else
                      if (newMaxC>maximumLimitValues[2])  { newMaxC=maximumLimitValues[2];  }
                    }

  //---------------------------------------------------------------------------
  minimumLimitValues[0]=newMinA;
  minimumLimitValues[1]=newMinB;
  minimumLimitValues[2]=newMinC;
  //---------------------------------------------------------------------------
  maximumLimitValues[0]=newMaxA;
  maximumLimitValues[1]=newMaxB;
  maximumLimitValues[2]=newMaxC;
}


int weAreAtALocalOptimum(
                         struct ikProblem * problem,
                         unsigned int chainID,
                         unsigned int partID,
                         float d,
                         float initialLoss,
                         float currentValues[3],
                         float delta[3],
                         unsigned int mIDS[3],
                         unsigned int verbose,
                         int penalizeSymmetryIn
                        )
{
  assert(d!=0.0);
  //Are we at a global optimum? -------------------------------------------------------------------
  int badLosses=0;
  for (unsigned int i=0; i<3; i++)
        {
            float rememberOriginalValue =  problem->chain[chainID].currentSolution->motion[mIDS[i]];
            problem->chain[chainID].currentSolution->motion[mIDS[i]] = currentValues[i]+d;
            float lossPlusD=calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/);
            problem->chain[chainID].currentSolution->motion[mIDS[i]] = currentValues[i]-d;
            float lossMinusD=calculateChainLoss(problem,chainID,partID,penalizeSymmetryIn,1/*Be economic*/);
            problem->chain[chainID].currentSolution->motion[mIDS[i]] = rememberOriginalValue;

            if ( (initialLoss<=lossPlusD) && (initialLoss<=lossMinusD) )
            {
                delta[i] = -d/10;  //very slight gradient // Why d ? and not 0
                ++badLosses;
            }
            else if ( (lossPlusD<initialLoss) && (lossPlusD<=lossMinusD) )
            {
                delta[i] = d;
            }
            else if ( (lossMinusD<initialLoss) && (lossMinusD<=lossPlusD) )
            {
                delta[i] = -d;
            }
            else if (verbose)
            {
                fprintf(stderr,RED "Dont know what to do with #%u value ..\n" NORMAL,i);
                fprintf(stderr,"-d = %0.2f,   +d = %0.2f, original = %0.2f\n",lossMinusD,lossPlusD,initialLoss);
                delta[i] = -d/10;  //very slight gradient // Why d ? and not 0
                ++badLosses;
            }
        }
  //-------------------------------------------------------------------------------------------------------
  return (badLosses==3);
}



int limitDeltasToThreshold(
                           float * delta,
                           float gradientExplosionThreshold
                          )
{
 int triggered = 0;
 if (delta[0]>0.0) { delta[0]=fmin(fabs(delta[0]), gradientExplosionThreshold); triggered=1; } else
                   { delta[0]=fmax(fabs(delta[0]),-gradientExplosionThreshold); triggered=1; }
 //-----------------------------------------------------------------------------------
 if (delta[1]>0.0) { delta[1]=fmin(fabs(delta[1]), gradientExplosionThreshold); triggered=1; } else
                   { delta[1]=fmax(fabs(delta[1]),-gradientExplosionThreshold); triggered=1; }
 //-----------------------------------------------------------------------------------
 if (delta[2]>0.0) { delta[2]=fmin(fabs(delta[2]), gradientExplosionThreshold); triggered=1; } else
                   { delta[2]=fmax(fabs(delta[2]),-gradientExplosionThreshold); triggered=1; }
 return triggered;
}




float iteratePartLoss(
                      struct ikProblem * problem,
                      struct ikConfiguration * config,
                      unsigned int iterationID,
                      unsigned int chainID,
                      unsigned int partID
                     )
{
    unsigned long startTime=0;
    //-----------------------------------------------------------------------------
    float minimumLossDeltaFromBestToBeAcceptable = config->eopchMinimumLossDelta; //Just be better than best..
    unsigned int maximumConsecutiveBadEpochs     = config->epochEarlyStopping;
    float lr                                     = config->learningRate;
    float learningRateDecayRate                  = config->learningRateDecayRate;
    float maximumAcceptableStartingLoss          = config->maximumAcceptableStartingLoss;
    float momentum                               = config->hcdMomentum;
    unsigned int epochs                          = config->epochs;
    unsigned int tryMaintainingLocalOptima       = config->tryMaintainingLocalOptima;
    float spring                                 = config->spring;
    float gradientExplosionThreshold             = config->gradientExplosionThreshold;
    char useSolutionHistory                      = !config->dontUseSolutionHistory;
    float useLangevinDynamics                    = config->useLangevinDynamics;
    unsigned int verbose                         = config->verbose;
    int PENALIZE_SYMMETRY_HEURISTIC              = config->penalizeSymmetriesHeuristic;
    //-----------------------------------------------------------------------------
    //Sensible Defaults
    if  (learningRateDecayRate==0.0)     { learningRateDecayRate = (float) 0.8; }
    if  (maximumConsecutiveBadEpochs==0) { maximumConsecutiveBadEpochs=1; } //By default 3
    if  (momentum==0.0)                  { momentum = (float) 0.4; } // Momentum | 0.9 Large / 0.2 Small
    //-----------------------------------------------------------------------------

    if (verbose) { startTime = GetTickCountMicrosecondsIK(); }

    //Modifiers in case of magnitude differences in problem chain
    //---------------------------------------------------------------
    if (problem->chain[chainID].part[partID].bigChanges)
         {
          lr=lr*10;
          learningRateDecayRate=learningRateDecayRate/2;
          gradientExplosionThreshold=gradientExplosionThreshold*10;
         } else
    if (problem->chain[chainID].part[partID].smallChanges)
         {
           lr=lr/10;
           learningRateDecayRate=learningRateDecayRate*2;
           gradientExplosionThreshold=gradientExplosionThreshold/10;
         }
    //---------------------------------------------------------------

    unsigned int numberOfMIDElements = 1 + problem->chain[chainID].part[partID].mIDEnd - problem->chain[chainID].part[partID].mIDStart;
    if (numberOfMIDElements!=3)
    {
       fprintf(stderr,RED "iteratePartLoss: %s Only 3 elements acceptable( got %u @ chain %u / part %u ) ..\n" NORMAL,problem->problemDescription,numberOfMIDElements,chainID,partID);
       fprintf(stderr,RED "mIDStart: %u\n" NORMAL,problem->chain[chainID].part[partID].mIDStart);
       fprintf(stderr,RED "mIDEnd: %u\n" NORMAL,problem->chain[chainID].part[partID].mIDEnd);
       fprintf(stderr,RED "forcing 3 elements from %u -> %u\n" NORMAL,problem->chain[chainID].part[partID].mIDStart,problem->chain[chainID].part[partID].mIDStart+2);
    }


    //Motion IDs so that we don't have to seek them in the problem struct every time they will be needed
    unsigned int mIDS[3] =
    {
        problem->chain[chainID].part[partID].mIDStart,   //This is ok because we have checked for 3 elements above
        problem->chain[chainID].part[partID].mIDStart+1, //This is ok because we have checked for 3 elements above
        problem->chain[chainID].part[partID].mIDStart+2  //This is ok because we have checked for 3 elements above
    };


    //---------------------------------------------------------------------------------------
    char limitsEngaged = problem->chain[chainID].part[partID].limits;
    //---------------------------------------------------------------------------------------
    float minimumLimitValues[3] = { problem->chain[chainID].part[partID].minimumLimitMID[0],
                                    problem->chain[chainID].part[partID].minimumLimitMID[1],
                                    problem->chain[chainID].part[partID].minimumLimitMID[2] };
    float maximumLimitValues[3] = { problem->chain[chainID].part[partID].maximumLimitMID[0],
                                    problem->chain[chainID].part[partID].maximumLimitMID[1],
                                    problem->chain[chainID].part[partID].maximumLimitMID[2] };
    //---------------------------------------------------------------------------------------
    //The original values we want to improve
    float originalValues[3] =     {
                                    problem->chain[chainID].currentSolution->motion[mIDS[0]],
                                    problem->chain[chainID].currentSolution->motion[mIDS[1]],
                                    problem->chain[chainID].currentSolution->motion[mIDS[2]]
                                  };
    //---------------------------------------------------------------------------------------
    //NEW functionality refuse to accept off-limit input..!
    if (limitsEngaged)
           {
             ensureValuesInLimits(originalValues,minimumLimitValues,maximumLimitValues);
             problem->chain[chainID].currentSolution->motion[mIDS[0]] = originalValues[0];
             problem->chain[chainID].currentSolution->motion[mIDS[1]] = originalValues[1];
             problem->chain[chainID].currentSolution->motion[mIDS[2]] = originalValues[2];
           }
    //---------------------------------------------------------------------------------------

    //The original values we want to improve
    unsigned int weHaveAPreviousSolutionHistory=(problem->previousSolution!=0);
    if (!useSolutionHistory) { weHaveAPreviousSolutionHistory=0; }

    float previousSolution[3] =
    {
        originalValues[0],
        originalValues[1],
        originalValues[2]
    };

    if (weHaveAPreviousSolutionHistory)
    {
        previousSolution[0] = problem->previousSolution->motion[mIDS[0]];
        previousSolution[1] = problem->previousSolution->motion[mIDS[1]];
        previousSolution[2] = problem->previousSolution->motion[mIDS[2]];
        if (limitsEngaged)
           {
             ensureValuesInLimits(previousSolution,minimumLimitValues,maximumLimitValues);
           }
    }



    //Shorthand to access joint ID and joint Name witout having to traverse the problem
    unsigned int jointID   = problem->chain[chainID].part[partID].jID;
    const char * jointName = problem->mc->jointHierarchy[jointID].jointName;
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------

    //Our armature has 500 d.o.f, if we do a calculateChainLoss this will calculate each and every one of them!!
    //Obviously we want to be really fast so we can't afford this, in order to speed up computations we will need to transform all parent joints
    //until the end joint of our chain...  iterateChainLoss
    float initialLoss = calculateChainLoss(
                                            problem,
                                            chainID,
                                            partID,
                                            PENALIZE_SYMMETRY_HEURISTIC,
                                            0//VERY IMPORTANT FOR THE FIRST CHAIN CALCULATION TO BE Non Economic otherwise subsequent calls will fail..
                                          );

    ///Having calculated all these joints from here on we only need to update this joint and its children ( we dont care about their parents since they dont change .. )
    bvh_markJointAsUsefulAndParentsAsUselessInTransform(problem->mc,&problem->chain[chainID].current2DProjectionTransform,jointID);
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------


   //Some reasons not to perform optimization is starting from NaN, starting from 0 or starting with a very high loss
   //---------------------------------------------------------------------------------------------------------------------------------------------------------------
   unsigned int initialLossIsNaN  = (initialLoss!=initialLoss);
   unsigned int initialLossIsZero = (initialLoss==0.0);
   //---------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (problem->chain[chainID].part[partID].endEffector)
        {
          fprintf(stderr,RED "What are we doing iteratePartLoss of endEffector..\n" NORMAL);
          return initialLoss;
        }

   if (initialLossIsNaN)
   {
       ++problem->chain[chainID].encounteredNumberOfNaNsAtStart;
       if(verbose)
         {
           fprintf(stderr,RED "Started with a NaN loss while processing chain %u for joint %s \n" NORMAL,chainID,jointName);
           bvh_printNotSkippedJoints(problem->mc,&problem->chain[chainID].current2DProjectionTransform);
         }
       return initialLoss;
   }
      else
   if (initialLossIsZero)
   {
        //If our loss is perfect we obviously can't improve it..
        if (verbose)
               { fprintf(stderr, GREEN"\nWon't optimize %s,  already perfect\n" NORMAL,jointName); }

        return initialLoss;
   }
       else
   if (maximumAcceptableStartingLoss>0.0)
   {
        //The positional subproblem gets a pass to help the other joints..
        int isItThePositionalSubproblem = ( (partID==0) && ( (chainID==0) || chainID==1) );

        //If we are really.. really.. far from the solution we don't want to try and do IK
        //as it will improve loss but may lead to a weird and incorrect pose
        if ( (initialLoss>maximumAcceptableStartingLoss) && (!isItThePositionalSubproblem) )
        {
            //We won't process a chain that is not the positional chain and is further than our maximum acceptable starting loss
            if (verbose)
                    { fprintf( stderr, RED"\nWon't optimize %s,  exceeded maximum acceptable starting loss by %0.2f%%\n" NORMAL,jointName, ((float) 100*initialLoss/maximumAcceptableStartingLoss) ); }

            return initialLoss;
        }
   } //Careful dont add another else here..
 //------------------------


    if (verbose)
          { fprintf(stderr,"\nOptimizing %s (initial loss %0.2f, iteration %u , chain %u, part %u)\n",jointName,initialLoss,iterationID,chainID,partID); }

//-------------------------------------------
//-------------------------------------------
//-------------------------------------------
//We only need to do this on the first iteration
//We dont want to constantly overwrite values with previous solutions and sabotage next iterations
if (iterationID==0)
{
    //If the previous solution is not given then it is impossible to use it
    if  ( (problem->previousSolution!=0) && (problem->previousSolution->motion!=0) )
    {
           //We need to remember the initial solution we where given
            float rememberInitialSolution[3]={
                                              problem->chain[chainID].currentSolution->motion[mIDS[0]],
                                              problem->chain[chainID].currentSolution->motion[mIDS[1]],
                                              problem->chain[chainID].currentSolution->motion[mIDS[2]]
                                             };

            //Maybe previous solution is closer to current observation?
            problem->chain[chainID].currentSolution->motion[mIDS[0]] = (float) problem->previousSolution->motion[mIDS[0]];
            problem->chain[chainID].currentSolution->motion[mIDS[1]] = (float) problem->previousSolution->motion[mIDS[1]];
            problem->chain[chainID].currentSolution->motion[mIDS[2]] = (float) problem->previousSolution->motion[mIDS[2]];
            float previousLoss = calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);

            if (previousLoss<initialLoss)
            {
                //Congratulations! better solution for free!
                 if (verbose)
                  { fprintf(stderr,GREEN "Previous solution for joint %s loss (%0.2f) is better than current (%0.2f) \n" NORMAL,jointName,previousLoss,initialLoss); }
                originalValues[0] = problem->chain[chainID].currentSolution->motion[mIDS[0]];
                originalValues[1] = problem->chain[chainID].currentSolution->motion[mIDS[1]];
                originalValues[2] = problem->chain[chainID].currentSolution->motion[mIDS[2]];
                //lr/=10;
                initialLoss = previousLoss;
            } else
            {
               //Previous solution is a worse solution,  let's forget about it and revert back!
               problem->chain[chainID].currentSolution->motion[mIDS[0]] = rememberInitialSolution[0];
               problem->chain[chainID].currentSolution->motion[mIDS[1]] = rememberInitialSolution[1];
               problem->chain[chainID].currentSolution->motion[mIDS[2]] = rememberInitialSolution[2];
            }
    }
}
//-------------------------------------------
//-------------------------------------------
    if ( problem->chain[chainID].part[partID].maeDeclared )
    {
      updateLimitsBasedOnMAE(
                             problem,
                             chainID,
                             partID,
                             originalValues,
                             minimumLimitValues,
                             maximumLimitValues
                            );
    }
    //----------------------------------------------------------------------------------
    float previousValues[3] = { originalValues[0],originalValues[1],originalValues[2] };
    float currentValues[3]  = { originalValues[0],originalValues[1],originalValues[2] };
    float bestValues[3]     = { originalValues[0],originalValues[1],originalValues[2] };
    //----------------------------------------------------------------------------------
    float previousLoss[3]   = { initialLoss, initialLoss, initialLoss };
    float currentLoss[3]    = { initialLoss, initialLoss, initialLoss };
    float previousDelta[3]  = { 0.0,0.0,0.0 };
    float gradient[3]       = { 0.0,0.0,0.0 };
    //----------------------------------------------------------------------------------
    float bestLoss = initialLoss;
    float loss     = initialLoss;

   ///--------------------------------------------------------------------------------------------------------------
   ///--------------------------------------------------------------------------------------------------------------
    float e=0.000001; //Prevent division by zero
    unsigned int consecutiveBadSteps=0; //Bad step counter
   ///--------------------------------------------------------------------------------------------------------------
    float delta[3]= {lr,lr,lr}; //Gives a positive initial direction..
   ///--------------------------------------------------------------------------------------------------------------
    if (tryMaintainingLocalOptima)
    {
        if (
             weAreAtALocalOptimum(
                                  problem,
                                  chainID,
                                  partID,
                                  lr,
                                  initialLoss,
                                  currentValues,
                                  delta,
                                  mIDS,
                                  verbose,
                                  PENALIZE_SYMMETRY_HEURISTIC
                                 )
            ) { return initialLoss; }
    } // tryMaintainingLocalOptima
   ///--------------------------------------------------------------------------------------------------------------
   ///--------------------------------------------------------------------------------------------------------------
    if (verbose)
    {
        fprintf(stderr,"  State |   loss   | rX  |  rY  |  rZ \n");
        fprintf(stderr,"Initial | %0.1f | %0.2f  |  %0.2f  |  %0.2f \n",initialLoss,originalValues[0],originalValues[1],originalValues[2]);
    }
   ///--------------------------------------------------------------------------------------------------------------
   ///--------------------------------------------------------------------------------------------------------------
    unsigned int executedEpochs=epochs;
    for (unsigned int currentEpoch=0; currentEpoch<epochs; currentEpoch++)
    {
        #define DO_FULL_ROLLBACK_LOSSES 0
        #define DO_COMBINED_LOSS 1

        #if DO_FULL_ROLLBACK_LOSSES
        //Calculate losses
        //  Spring to initial -> distanceFromInitial=fabs(currentValues[0..2] - originalValues[0..2]);
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        problem->chain[chainID].currentSolution->motion[mIDS[0]] = currentValues[0];
        currentLoss[0]=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);// + spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss[0]>bestLoss) { problem->chain[chainID].currentSolution->motion[mIDS[0]] = previousValues[0]; }
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        problem->chain[chainID].currentSolution->motion[mIDS[1]] = currentValues[1];
        currentLoss[1]=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);// + spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss[1]>bestLoss) { problem->chain[chainID].currentSolution->motion[mIDS[1]] = previousValues[1]; }
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        problem->chain[chainID].currentSolution->motion[mIDS[2]] = currentValues[2];
        currentLoss[2]=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);// + spring * distanceFromInitial * distanceFromInitial;
        if (currentLoss[2]>bestLoss) { problem->chain[chainID].currentSolution->motion[mIDS[2]] = previousValues[2]; }
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        #else
         // Calculate losses only when needed
         // This logic avoids *two thirds* of calculateChainLoss codes and thus uses half the number of 4x4 Matrix Multiplications!
         // It is a big improvement doing the same work as the old code segment above..!
         //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
         currentLoss[0] = loss;
         if (currentLoss[0]>bestLoss) {
                                        problem->chain[chainID].currentSolution->motion[mIDS[0]] = previousValues[0];
                                        currentLoss[0] = calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);
                                        loss = currentLoss[0];
                                      }
         //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
         currentLoss[1] = loss;
         if (currentLoss[1]>bestLoss) {
                                        problem->chain[chainID].currentSolution->motion[mIDS[1]] = previousValues[1];
                                        currentLoss[1]=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);
                                        loss = currentLoss[1];
                                      }
         //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
         currentLoss[2] = loss;
         if (currentLoss[2]>bestLoss) {
                                        problem->chain[chainID].currentSolution->motion[mIDS[2]] = previousValues[2];
                                        currentLoss[2]=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);
                                        loss = currentLoss[2];
                                      }
         //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        #endif // DO_FULL_ROLLBACK_LOSSES


        //We multiply by 0.5 to do a "One Half Mean Squared Error"
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        previousDelta[0] = delta[0];
        gradient[0]      = (float) 0.5 * (previousLoss[0] - currentLoss[0]) / (delta[0]+e);
        delta[0]         = momentum * delta[0] + (float) lr * gradient[0];
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        previousDelta[1] = delta[1];
        gradient[1]      = (float) 0.5 * (previousLoss[1] - currentLoss[1]) / (delta[1]+e);
        delta[1]         = momentum * delta[1] + (float) lr * gradient[1];
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
        previousDelta[2] = delta[2];
        gradient[2]      = (float) 0.5 * (previousLoss[2] - currentLoss[2]) / (delta[2]+e);
        delta[2]         = momentum * delta[2] + (float) lr * gradient[2];
        //-------------------  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------

        if (useLangevinDynamics!=0.0)
        {
           //Attempt to use Langevin dynamics for annealed gradient descent
           delta[0]+=randomNoise(useLangevinDynamics);
           delta[1]+=randomNoise(useLangevinDynamics);
           delta[2]+=randomNoise(useLangevinDynamics);
        }

        //Safeguard against gradient explosions which we detect when we see large gradients
        unsigned int deltaExploded       = ( (fabs(delta[0])>gradientExplosionThreshold) || (fabs(delta[1])>gradientExplosionThreshold) || (fabs(delta[2])>gradientExplosionThreshold) );
        unsigned int encounteredNaNDelta = ( (delta[0]!=delta[0]) || (delta[1]!=delta[1]) || (delta[2]!=delta[2]) );

        if  (encounteredNaNDelta)
        {
            ++problem->chain[chainID].encounteredExplodingGradients;
            fprintf(stderr,RED "EXPLODED %s @ %u/%u | d{%0.1f,%0.1f,%0.1f}/%0.1f | mIDS{%u,%u,%u}\n" NORMAL,jointName,currentEpoch,epochs,delta[0],delta[1],delta[2],gradientExplosionThreshold,mIDS[0],mIDS[1],mIDS[2]);
            if (verbose)
            {
             fprintf(stderr,RED "previousDeltas[%0.2f,%0.2f,%0.2f]\n" NORMAL,previousDelta[0],previousDelta[1],previousDelta[2]);
             fprintf(stderr,RED "currentDeltas[%0.2f,%0.2f,%0.2f]\n" NORMAL,delta[0],delta[1],delta[2]);
             fprintf(stderr,RED "gradients[%0.2f,%0.2f,%0.2f]\n" NORMAL,gradient[0],gradient[1],gradient[2]);
             fprintf(stderr,RED "previousLoss[%0.2f,%0.2f,%0.2f]\n" NORMAL,previousLoss[0],previousLoss[1],previousLoss[2]);
             fprintf(stderr,RED "currentLoss[%0.2f,%0.2f,%0.2f]\n" NORMAL,currentLoss[0],currentLoss[1],currentLoss[2]);
             fprintf(stderr,RED "lr = %f momentum = %0.2f \n" NORMAL,lr,momentum);
            }
             //Just stop after an explosion..
            executedEpochs=currentEpoch;
            break;
        } else
        if  (deltaExploded)
        {
            ++problem->chain[chainID].encounteredExplodingGradients;
            fprintf(stderr,RED "SUPPRESSED EXPLOSION %s @ %u/%u | d{%0.1f,%0.1f,%0.1f}/%0.1f | mIDS{%u,%u,%u}\n" NORMAL,jointName,currentEpoch,epochs,delta[0],delta[1],delta[2],gradientExplosionThreshold,mIDS[0],mIDS[1],mIDS[2]);
            limitDeltasToThreshold(delta,gradientExplosionThreshold);
             //Just stop after an explosion..
            executedEpochs=currentEpoch;
            break;
        }


        //----------------------------------------------
        //Remember previous loss/values
        //----------------------------------------------
        previousLoss[0]=currentLoss[0];
        previousLoss[1]=currentLoss[1];
        previousLoss[2]=currentLoss[2];
        //----------------------------------------------
        previousValues[0]=currentValues[0];
        previousValues[1]=currentValues[1];
        previousValues[2]=currentValues[2];
        //----------------------------------------------
        //We advance our current state..
        //----------------------------------------------
        currentValues[0]+=delta[0];
        currentValues[1]+=delta[1];
        currentValues[2]+=delta[2];
        //----------------------------------------------
        if (limitsEngaged)
           { ensureValuesInLimits(currentValues,minimumLimitValues,maximumLimitValues); }
        //----------------------------------------------
        //We store our new values and calculate our new loss
        //----------------------------------------------
        problem->chain[chainID].currentSolution->motion[mIDS[0]] = currentValues[0];
        problem->chain[chainID].currentSolution->motion[mIDS[1]] = currentValues[1];
        problem->chain[chainID].currentSolution->motion[mIDS[2]] = currentValues[2];
        //----------------------------------------------
        #if DO_COMBINED_LOSS
          loss=calculateChainLoss(problem,chainID,partID,PENALIZE_SYMMETRY_HEURISTIC,1/*Be economic*/);
        #else
          //Just use final loss without reconfirming new delta limits
          loss=currentLoss[2];
        #endif
        //----------------------------------------------

        // If loss is NaN
        if (loss!=loss)
        {
            //Immediately terminate when encountering NaN, it will be a waste of resources otherwise
            if (verbose)
                    { fprintf(stderr,RED "%07u |NaN| %0.2f  |  %0.2f  |  %0.2f \n" NORMAL,currentEpoch,currentValues[0],currentValues[1],currentValues[2]); }
            executedEpochs=currentEpoch;
            break;
        } else
        if (loss + minimumLossDeltaFromBestToBeAcceptable < bestLoss)
        {
            //Loss has been improved..!
            bestLoss=loss;
            bestValues[0]=currentValues[0];
            bestValues[1]=currentValues[1];
            bestValues[2]=currentValues[2];
            consecutiveBadSteps=0;
            if (verbose)
                  { fprintf(stderr,"%07u | %0.1f | %0.2f(%0.2f)  |  %0.2f(%0.2f)  |  %0.2f(%0.2f) \n",currentEpoch,loss,currentValues[0],delta[0],currentValues[1],delta[1],currentValues[2],delta[2]); }
        }
        else
        { //Loss has not been improved..!
            ++consecutiveBadSteps;
            if (verbose)
                  { fprintf(stderr,YELLOW "%07u | %0.1f | %0.2f(%0.2f)  |  %0.2f(%0.2f)  |  %0.2f(%0.2f) \n" NORMAL,currentEpoch,loss,currentValues[0],delta[0],currentValues[1],delta[1],currentValues[2],delta[2]); }
        }

        if (consecutiveBadSteps>=maximumConsecutiveBadEpochs)
        {
            if (verbose)
                 { fprintf(stderr,YELLOW "Early Stopping\n" NORMAL); }
            executedEpochs=currentEpoch;
            break;
        }
        //----------------------------------------------
        lr = (float) learningRateDecayRate * lr;
        //Gradual fine tuning.. On a first glance it works worse..
        //lr = lr * (learningRateDecayRate * iterationID); <- could also geometrically scale
        momentum = momentum * 0.95; // Scale momentum!
        //----------------------------------------------
    } // for number of epochs


    if (verbose)
    {   unsigned long endTime = GetTickCountMicrosecondsIK();
        fprintf(stderr,"Optimization for joint %s \n", jointName);
        fprintf(stderr,"Improved loss from %0.2f to %0.2f ( %0.2f%% ) in %lu microseconds \n",initialLoss,bestLoss, 100 - ( (float) 100* bestLoss/initialLoss ),endTime-startTime);
        fprintf(stderr,"Optimized values changed from %0.2f,%0.2f,%0.2f to %0.2f,%0.2f,%0.2f\n",originalValues[0],originalValues[1],originalValues[2],bestValues[0],bestValues[1],bestValues[2]);
        fprintf(stderr,"correction of %0.2f,%0.2f,%0.2f deg\n",bestValues[0]-originalValues[0],bestValues[1]-originalValues[1],bestValues[2]-originalValues[2]);
        fprintf(stderr,"correction rate of %0.2f,%0.2f,%0.2f deg\n",(bestValues[0]-originalValues[0])/executedEpochs,(bestValues[1]-originalValues[1])/executedEpochs,(bestValues[2]-originalValues[2])/executedEpochs);
    }

    if (weHaveAPreviousSolutionHistory)
    {
      if (
           examineSolutionAndKeepIfItIsBetter(
                                              problem,
                                              iterationID,
                                              chainID,
                                              partID,
                                              mIDS,
                                              originalValues,
                                              bestValues,
                                              &bestLoss,
                                              spring,
                                              //-------------------------
                                              previousSolution,
                                              //-------------------------
                                              PENALIZE_SYMMETRY_HEURISTIC
                                            )
         )
         {
            //fprintf(stderr,"Optimization for joint %s rolled back\n",jointName);
            ++problem->chain[chainID].encounteredWorseSolutionsThanPrevious;
         }
    }

    if (limitsEngaged)
        { ensureValuesInLimits(bestValues,minimumLimitValues,maximumLimitValues); }

    //After finishing with the optimization procedure we store the best result we achieved..!
    problem->chain[chainID].currentSolution->motion[mIDS[0]] = bestValues[0];
    problem->chain[chainID].currentSolution->motion[mIDS[1]] = bestValues[1];
    problem->chain[chainID].currentSolution->motion[mIDS[2]] = bestValues[2];

    //Multi threaded code (and single threaded code) needs to also concurrently update the final solution for next iterations
    //In order to get rid of extra bureocracies with motion buffers we also perform the update here..
    //Only a chain is responsible and should update the specific motion values anyways
    problem->currentSolution->motion[mIDS[0]] = bestValues[0];
    problem->currentSolution->motion[mIDS[1]] = bestValues[1];
    problem->currentSolution->motion[mIDS[2]] = bestValues[2];

    //Track losses
    //--------------------------------------------------------------------------
    problem->chain[chainID].previousLoss = problem->chain[chainID].currentLoss;
    problem->chain[chainID].currentLoss  = bestLoss;
    if (problem->chain[chainID].lossUpdates==0)
         { problem->chain[chainID].initialLoss = problem->chain[chainID].currentLoss; }
    problem->chain[chainID].lossUpdates = problem->chain[chainID].lossUpdates+1;
    //--------------------------------------------------------------------------

    return bestLoss;
}

int iterateChainLoss(
                     struct ikProblem * problem,
                     struct ikConfiguration * config,
                     unsigned int iterationID,
                     unsigned int chainID
                    )
{
    problem->chain[chainID].status = BVH_IK_STARTED;

     //Make sure chain has been fully extended to root joint..
    bvh_markAllJointsAsUselessInTransform(problem->mc,&problem->chain[chainID].current2DProjectionTransform);
    for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
    {
      unsigned int jointID = problem->chain[chainID].part[partID].jID;
      bvh_markJointAndParentsAsUsefulInTransform(problem->mc,&problem->chain[chainID].current2DProjectionTransform,jointID);
    }

    for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
    {
        if (!problem->chain[chainID].part[partID].endEffector)
        { //If the part is  not an end effector it has parameters to change and improve
          iteratePartLoss(
                          problem,
                          config,
                          iterationID,
                          chainID,
                          partID
                         );
         }
    }

    problem->chain[chainID].status = BVH_IK_FINISHED_ITERATION;

    return 1;
}

float calculateCachedProblemLoss(struct ikProblem * problem)
{
 float res=0.0;
 for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
 {
    res+=problem->chain[chainID].currentLoss;
 }
 return res;
}

//This is the regular and easy to follow serial implementation where for each iteration we go through
//each one of the chains in order.. We still mark the chain status to ensure 1:1 operation with the multithreaded
//version of the code..
int singleThreadedSolver(
                         struct ikProblem * problem,
                         struct ikConfiguration * ikConfig
                        )
{
  for (unsigned int iterationID=0; iterationID<ikConfig->iterations; iterationID++)
    {
        //-----------------------------------------------------------------------
        float lossBeforeOptimization = calculateCachedProblemLoss(problem);
        //-----------------------------------------------------------------------
        for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
        {
            //Before we start we will make a copy of the problem->currentSolution to work on improving it..
            copyMotionBuffer(problem->chain[chainID].currentSolution,problem->currentSolution);

            problem->chain[chainID].currentIteration=iterationID;
            iterateChainLoss(
                              problem,
                              ikConfig,
                              iterationID,
                              chainID
                            );

             //Each iteratePartLoss call updates the problem->currentSolution with the latest and greatest solution
             //If we are here it means problem->currentSolution has the best solution IK could find..
        }
        //-----------------------------------------------------------------------
        float lossAfterOptimization = calculateCachedProblemLoss(problem);
        //-----------------------------------------------------------------------
        if ( (iterationID>1) && (ikConfig->iterationEarlyStopping) )
        {
            if (lossBeforeOptimization-lossAfterOptimization<=ikConfig->iterationMinimumLossDelta)
            { //Optimization was ineffective! We are probably dealing with ill-conditioned problem
                if (ikConfig->verbose)
                  {
                    fprintf(stderr,RED "Terminating on %u/%u iterations to save cycles \n" NORMAL,iterationID,ikConfig->iterations);
                    fprintf(stderr,RED "Loss went %0.2f -> %0.2f  (Limit is %0.2f) \n" NORMAL,lossBeforeOptimization,lossAfterOptimization,ikConfig->iterationMinimumLossDelta);
                  }
                break;
            } else
            {
              if (ikConfig->verbose)
                  {
                    fprintf(stderr,GREEN "%u/%u iterations \n" NORMAL,iterationID,ikConfig->iterations);
                    fprintf(stderr,NORMAL "Loss went %0.2f -> %0.2f\n" NORMAL,lossBeforeOptimization,lossAfterOptimization);
                  }
            }
        }
        //-----------------------------------------------------------------------
    }

   //We are running using a single thread so we mark everything finished at the same time..
   for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
        {
           problem->chain[chainID].status = BVH_IK_FINISHED_EVERYTHING;
        }
   return 1;
}

///=====================================================================================
///=====================================================================================
///=====================================================================================
///                             Start of Multi Threaded Code
///=====================================================================================
///=====================================================================================
///=====================================================================================

void * iterateChainLossWorkerThread(void * arg)
{
  //We are a thread so lets retrieve our variables..
  struct threadContext * ptr = (struct threadContext *) arg;
  fprintf(stderr,"Thread-%u: Started..!\n",ptr->threadID);
  struct passIKContextToThread * contextArray = (struct passIKContextToThread *) ptr->argumentToPass;
  struct passIKContextToThread * ctx = &contextArray[ptr->threadID];
  fprintf(stderr,"Chain-%u: Started..!\n",ctx->chainID);

  threadpoolWorkerInitialWait(ptr);

  while (threadpoolWorkerLoopCondition(ptr))
  {
    //fprintf(stderr,"Thread %u doing actual Work\n",ctx->threadID);
    //--------------------------------
         iterateChainLoss(
                           ctx->problem,
                           ctx->ikConfig,
                           ctx->problem->chain[ctx->chainID].currentIteration,
                           ctx->chainID
                         );
    //--------------------------------
    threadpoolWorkerLoopEnd(ptr);
  }

  return 0;
}


//This is the multi threaded version of the code..!
//This is more complex than just spawning one thread per problem because we have to ensure that certain chains get processed in certain order
int multiThreadedSolver(
                        struct ikProblem * problem,
                        struct ikConfiguration * ikConfig
                       )
{
  //fprintf(stderr,"multiThreadedSolver called\n");
  //unsigned int numberOfFreshlySpawnThreads=0;
  unsigned int numberOfWorkerThreads = 0;

  //Make sure all threads needed are created but only paying the cost of creating a thread once..!
  for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
        {
          if ( problem->chain[chainID].parallel )
              {
                 problem->workerContext[numberOfWorkerThreads].problem=problem;
                 problem->workerContext[numberOfWorkerThreads].ikConfig=ikConfig;
                 problem->workerContext[numberOfWorkerThreads].chainID=chainID;
                 problem->workerContext[numberOfWorkerThreads].threadID=numberOfWorkerThreads;
                 ++numberOfWorkerThreads;
              }
        }

   if (!problem->threadPool.initialized)
   {
     if (
         !threadpoolCreate(
                            &problem->threadPool,
                            numberOfWorkerThreads,
                            (void*) iterateChainLossWorkerThread,
                            (void*) &problem->workerContext
                          )
        )
     {
        return 0;
     }
    fprintf(stderr,"Survived threadpool creation \n");
    sleep(1);
   }
  //fprintf(stderr,GREEN "Worker Threads =%u / Freshly spawned threads = %u \n" NORMAL,numberOfWorkerThreads,numberOfFreshlySpawnThreads);

  //We will perform a number of iterations  each of which have to be synced in the end..
  for (unsigned int iterationID=0; iterationID<ikConfig->iterations; iterationID++)
    {
        //-----------------------------------------------------------------------
        float lossBeforeOptimization = calculateCachedProblemLoss(problem);
        //-----------------------------------------------------------------------
        threadpoolMainThreadPrepareWorkForWorkers(&problem->threadPool);

        //We go through each chain, if the chain is single threaded we do the same as the singleThreadedSolver
        //if the thread is parallel then we just ask it to start processing the current data and we then need to stop and wait to gather results..
        for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
        {
              //Before we start we will make a copy of the problem->currentSolution to work on improving it..
              copyMotionBuffer(problem->chain[chainID].currentSolution,problem->currentSolution);

              problem->chain[chainID].currentIteration=iterationID;
              if (!problem->chain[chainID].parallel)
              {  //Normal chains run normally..
                //fprintf(stderr,"Running single threaded task for chain %u\n",chainID);
                iterateChainLoss(
                                 problem,
                                 ikConfig,
                                 iterationID,
                                 chainID
                                );

                  //Each iteratePartLoss call updates the problem->currentSolution with the latest and greatest solution
                  //If we are here it means problem->currentSolution has the best solution IK could find..
              }
        }

        threadpoolMainThreadWaitForWorkersToFinish(&problem->threadPool);
        //--------------------------------------------------

        //-----------------------------------------------------------------------
        float lossAfterOptimization = calculateCachedProblemLoss(problem);
        //-----------------------------------------------------------------------
        if ( (iterationID>1) && (ikConfig->iterationEarlyStopping) )
        {
            if (lossBeforeOptimization-lossAfterOptimization<=ikConfig->iterationMinimumLossDelta)
            { //Optimization was ineffective! We are probably dealing with ill-conditioned problem
                if (ikConfig->verbose)
                  {
                    fprintf(stderr,RED "Terminating on %u/%u iterations to save cycles \n" NORMAL,iterationID,ikConfig->iterations);
                    fprintf(stderr,RED "Loss went %0.2f -> %0.2f  (Limit is %0.2f) \n" NORMAL,lossBeforeOptimization,lossAfterOptimization,ikConfig->iterationMinimumLossDelta);
                  }
                break;
            } else
            {
              if (ikConfig->verbose)
                  {
                    fprintf(stderr,GREEN "%u/%u iterations \n" NORMAL,iterationID,ikConfig->iterations);
                    fprintf(stderr,NORMAL "Loss went %0.2f -> %0.2f\n" NORMAL,lossBeforeOptimization,lossAfterOptimization);
                  }
            }
        }
        //-----------------------------------------------------------------------
    }

  return 1;
}

///=====================================================================================
///=====================================================================================
///=====================================================================================
///                            End of Multi Threaded Code
///=====================================================================================
///=====================================================================================
///=====================================================================================

int extrapolateSolution(
                        struct MotionBuffer * a,
                        struct MotionBuffer * b,
                        struct MotionBuffer * extrapolated
                       )
{
    if ( (a!=0) && (b!=0) && (extrapolated!=0) )
        {
         if ( (a->motion!=0) && (b->motion!=0) && (extrapolated->motion!=0) )
          {
              //All buffers are there..
              for (unsigned int mID=0; mID<extrapolated->bufferSize; mID++)
              {
                  extrapolated->motion[mID] = b->motion[mID] + ( b->motion[mID] - a->motion[mID] );
              }
              return 1;
          }
        }
    return 0;
}





int ensureInitialPositionIsInFrustrum(
                                      struct ikProblem * problem,
                                      struct simpleRenderer *renderer,
                                      struct MotionBuffer * solution,
                                      struct MotionBuffer * previousSolution
                                     )
{
   if (renderer==0) { return 0; }
   if (solution==0) { return 0; }
   if (solution->motion==0) { return 0; }
   if (previousSolution==0) { return 0; }
   if (previousSolution->motion==0) { return 0; }

   float closestDistanceToCameraInCM=13; //30 cm


   if (problem->nearCutoffPlaneDeclared)
       {
           //Overriding with near cutoff plane declared in problem
           closestDistanceToCameraInCM = problem->nearCutoffPlane;
       }

   //TODO :
   //Ensure that  pose is not out of the bounds of camera ?
   //If it is inverse kinematics wont know what to do..
    if (solution->motion[2] > -1 * closestDistanceToCameraInCM)
    {
        fprintf(stderr,RED "Warning: Detected pose behind camera! ..\n" NORMAL);
        fprintf(stderr,"Initial X Y Z Position was (%0.2f,%0.2f,%0.2f) ! ..\n",solution->motion[0],solution->motion[1],solution->motion[2]);
        if ( (previousSolution!=0) && (previousSolution->motion!=0) )
        {
            if (previousSolution->motion[2] < -1 * closestDistanceToCameraInCM)
                    {
                        fprintf(stderr,GREEN "Fixed using previous frame ! ..\n" NORMAL);
                        solution->motion[0]=previousSolution->motion[0];
                        solution->motion[1]=previousSolution->motion[1];
                        /// This is the most important  --------------------------------
                        solution->motion[2]=previousSolution->motion[2];
                        ///-------------------------------------------------------------------------
                        solution->motion[3]=previousSolution->motion[3];
                        solution->motion[4]=previousSolution->motion[4];
                        solution->motion[5]=previousSolution->motion[5];
                    }
        }

        if (solution->motion[2] > -1 * closestDistanceToCameraInCM)
        {
                 solution->motion[2]=-130;
                 fprintf(stderr,RED "Warning: ensureInitialPositionIsInFrustrum will push solution back to %f ..\n" NORMAL,solution->motion[2]);
        }
        fprintf(stderr,"Initial X Y Z Position after tweak is (%0.2f,%0.2f,%0.2f) ! ..\n",solution->motion[0],solution->motion[1],solution->motion[2]);
    }

     return 1;
}



void compareChainsAndAdoptBest(
                               struct BVH_MotionCapture * mc,
                               struct simpleRenderer *renderer,
                               //---------------------------------
                               struct ikProblem * problem,
                               unsigned int startChain,
                               unsigned int endChain,
                               //---------------------------------
                               struct ikConfiguration * ikConfig,
                               //---------------------------------
                               struct MotionBuffer * currentSolution,
                               struct MotionBuffer * checkIfItIsBetterSolution,
                               //---------------------------------
                               struct BVH_Transform * bvhCurrentTransform,
                               struct BVH_Transform * bvhCheckIfItIsBetterTransform,
                               //---------------------------------
                               struct BVH_Transform * bvhTargetTransform
                               //---------------------------------
                              )
{
 //fprintf(stderr,"compareChainsAndAdoptBest started ");
 if ( (currentSolution!=0) && (currentSolution->motion!=0) && (checkIfItIsBetterSolution!=0) && (checkIfItIsBetterSolution->motion!=0) )
 {
  if ( (currentSolution->bufferSize==checkIfItIsBetterSolution->bufferSize) )
  {
    if ( (problem->numberOfChains>startChain) && (problem->numberOfChains>endChain) )
    {
      for (unsigned int chainID=startChain; chainID<endChain; chainID++)
                {
                  unsigned int samplesCurrent     = 0;
                  unsigned int samplesPrevious    = 0;
                  float currentSolutionChainLoss  = 0.0;
                  float previousSolutionChainLoss = 0.0;

                  unsigned int partIDStart = 0;
                  unsigned int failedProjections=0;
                  for (unsigned int partID=partIDStart; partID<problem->chain[chainID].numberOfParts; partID++)
                    {
                     unsigned int jID=problem->chain[chainID].part[partID].jID;
                     failedProjections += ( bvh_projectJIDTo2D(mc,bvhCurrentTransform,renderer,jID,0,0)  == 0 );
                     failedProjections += ( bvh_projectJIDTo2D(mc,bvhCheckIfItIsBetterTransform,renderer,jID,0,0) == 0 );

                     if (failedProjections==0)
                     {
                      jID=problem->chain[chainID].part[partID].jID;

                      ///Warning: When you change this please change meanBVH2DDistance as well!
                      float cX=(float) bvhCheckIfItIsBetterTransform->joint[jID].pos2D[0];
                      float cY=(float) bvhCheckIfItIsBetterTransform->joint[jID].pos2D[1];
                      float sX=(float) bvhCurrentTransform->joint[jID].pos2D[0];
                      float sY=(float) bvhCurrentTransform->joint[jID].pos2D[1];
                      float tX=(float) bvhTargetTransform->joint[jID].pos2D[0];
                      float tY=(float) bvhTargetTransform->joint[jID].pos2D[1];

                      //Only use source/target joints that exist and are not occluded..

                      if ((tX!=0.0) || (tY!=0.0))
                      {
                        //Don't do anything without the target point..
                        if ((sX!=0.0) || (sY!=0.0))
                        {
                          //Our current solution
                          currentSolutionChainLoss += getSquared2DPointDistance(sX,sY,tX,tY) * problem->chain[chainID].part[partID].jointImportance;
                          samplesCurrent = samplesCurrent + 1;
                        }

                        if ((cX!=0.0) || (cY!=0.0))
                        {
                          //The solution we want to check if is better
                          previousSolutionChainLoss += getSquared2DPointDistance(cX,cY,tX,tY) * problem->chain[chainID].part[partID].jointImportance;
                          samplesPrevious = samplesPrevious + 1;
                        }
                      }
                     }
                    }


                    if ( (samplesPrevious==0) || (samplesCurrent==0) )
                    {
                      fprintf(stderr,RED "compareChainsAndAdoptBest: Blocking chain %u update without samples..\n" NORMAL,chainID);
                    } else
                    if (currentSolutionChainLoss > previousSolutionChainLoss)
                    {
                        ++problem->chain[chainID].encounteredAdoptedBest;
                        if (ikConfig->verbose)
                             { fprintf(stderr,RED "compareChainsAndAdoptBest: Chain %u came out worse\n" NORMAL,chainID); }

                        for (unsigned int partID=partIDStart; partID<problem->chain[chainID].numberOfParts; partID++)
                        {
                         //Only perform this on non end-effector parts of chains
                         if (!problem->chain[chainID].part[partID].endEffector)
                         {
                         //----------------------------------------------------------------------------------
                         unsigned int mIDS[3] = {
                                                 problem->chain[chainID].part[partID].mIDStart,
                                                 problem->chain[chainID].part[partID].mIDStart+1,
                                                 problem->chain[chainID].part[partID].mIDStart+2
                                                };
                         //----------------------------------------------------------------------------------
                         if (
                              (mIDS[0]<currentSolution->bufferSize) &&
                              (mIDS[1]<currentSolution->bufferSize) &&
                              (mIDS[2]<currentSolution->bufferSize)
                           )
                              {
                                currentSolution->motion[mIDS[0]]=checkIfItIsBetterSolution->motion[mIDS[0]];
                                currentSolution->motion[mIDS[1]]=checkIfItIsBetterSolution->motion[mIDS[1]];
                                currentSolution->motion[mIDS[2]]=checkIfItIsBetterSolution->motion[mIDS[2]];
                              } else
                              {
                                viewProblem(problem);
                                fprintf(stderr,RED "BUG: compareChainsAndAdoptBest: Incorrect chain mIDS for chain %u part %u\n" NORMAL,chainID,partID);
                                fprintf(stderr,RED "mIDS[0]=%u mIDS[1]=%u mIDS[2]=%u\n" NORMAL,mIDS[0],mIDS[1],mIDS[2]);
                              }
                         //----------------------------------------------------------------------------------
                         }
                        } //part loop..
                    } //We found a better chain.. take it!
                } //chain loop..
  } else
  {
    fprintf(stderr,RED "compareChainsAndAdoptBest: Cannot operate on a chain [%u-%u] out of limits[%u]..\n" NORMAL,startChain,endChain,problem->numberOfChains);
  }
 } else
 {
   fprintf(stderr,"compareChainsAndAdoptBest: Cannot operate on different size chains..\n");
 }
 }
  //fprintf(stderr,"compareChainsAndAdoptBest survived ");
}



int ensureFinalProposedSolutionIsBetterInParts(
                                               struct BVH_MotionCapture * mc,
                                               struct simpleRenderer *renderer,
                                               //---------------------------------
                                               const char * label,
                                               //---------------------------------
                                               struct ikProblem * problem,
                                               unsigned int startChain,
                                               unsigned int endChain,
                                               //---------------------------------
                                               struct ikConfiguration * ikConfig,
                                               //---------------------------------
                                               struct MotionBuffer * currentSolution,
                                               struct MotionBuffer * previousSolution,
                                               //---------------------------------
                                               struct BVH_Transform * bvhTargetTransform
                                               //---------------------------------
                                              )
{
   if ( (currentSolution==0)  || (currentSolution->motion==0) )     { return 0; }
   if ( (previousSolution==0) || (previousSolution->motion==0) )    { return 0; }
   if (bvhTargetTransform==0) { return 0; }
   //fprintf(stderr,GREEN "ensureFinalProposedSolutionIsBetterInParts running for %s\n" NORMAL,label);
   //------------------------------------------------
   struct BVH_Transform bvhCurrentTransform  = {0};
   struct BVH_Transform bvhPreviousTransform = {0};
   //------------------------------------------------
   if (bvh_loadTransformForMotionBuffer(mc,currentSolution->motion,&bvhCurrentTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
           {
            if (bvh_loadTransformForMotionBuffer(mc,previousSolution->motion,&bvhPreviousTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
             {
               compareChainsAndAdoptBest(
                                         mc,
                                         renderer,
                                         problem,
                                         startChain,
                                         endChain,
                                         ikConfig,
                                         currentSolution,
                                         previousSolution,
                                         &bvhCurrentTransform,
                                         &bvhPreviousTransform,
                                         bvhTargetTransform
                                        );
               return 1;
             }
           }
    bvh_freeTransform(&bvhCurrentTransform);
    bvh_freeTransform(&bvhPreviousTransform);
   //------------------------------------------------
   return 0;
}

int springToZeroParts(
                       struct BVH_MotionCapture * mc,
                       struct simpleRenderer *renderer,
                       //---------------------------------
                       struct ikProblem * problem,
                       unsigned int startChain,
                       unsigned int endChain,
                       //---------------------------------
                       struct ikConfiguration * ikConfig,
                       //---------------------------------
                       struct MotionBuffer * currentSolution,
                       //---------------------------------
                       struct BVH_Transform * bvhTargetTransform
                       //---------------------------------
                    )
{
   if (currentSolution==0)    { return 0; }
   fprintf(stderr,GREEN "springToZeroParts running \n" NORMAL);
   //------------------------------------------------
   struct BVH_Transform bvhCurrentTransform  = {0};
   struct BVH_Transform bvhZeroTransform     = {0};
   //------------------------------------------------
   struct MotionBuffer * zeroSolution = mallocNewMotionBufferAndCopy(mc,currentSolution);
   //------------------------------------------------
   if (zeroSolution!=0)
   {
    if ( (currentSolution->bufferSize==zeroSolution->bufferSize) )
     {
      if ( (problem->numberOfChains>startChain) && (problem->numberOfChains>endChain) )
      {
        for (unsigned int chainID=startChain; chainID<endChain; chainID++)
                {
                  unsigned int partIDStart = 0;
                  for (unsigned int partID=partIDStart; partID<problem->chain[chainID].numberOfParts; partID++)
                    {
                         unsigned int mIDS[3] = {
                                                 problem->chain[chainID].part[partID].mIDStart,
                                                 problem->chain[chainID].part[partID].mIDStart+1,
                                                 problem->chain[chainID].part[partID].mIDStart+2
                                                };

                         currentSolution->motion[mIDS[0]]=0;
                         currentSolution->motion[mIDS[1]]=0;
                         currentSolution->motion[mIDS[2]]=0;
                    }
                }

     if (bvh_loadTransformForMotionBuffer(mc,currentSolution->motion,&bvhCurrentTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
           {
            if (bvh_loadTransformForMotionBuffer(mc,zeroSolution->motion,&bvhZeroTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
             {
               compareChainsAndAdoptBest(
                                         mc,
                                         renderer,
                                         problem,
                                         startChain,
                                         endChain,
                                         ikConfig,
                                         currentSolution,
                                         zeroSolution,
                                         &bvhCurrentTransform,
                                         &bvhZeroTransform,
                                         bvhTargetTransform
                                        );
               return 1;
             }
           }
    //------------------------------------------------
    freeMotionBuffer(&zeroSolution);
    zeroSolution=0;
   }
    }
   }
   return 0;
}

int doExtrapolatedGuess(
                        struct BVH_MotionCapture * mc,
                        struct simpleRenderer *renderer,
                        struct ikProblem * problem,
                        struct ikConfiguration * ikConfig,
                        //---------------------------------
                        struct MotionBuffer * solution,
                        struct MotionBuffer * penultimateSolution,
                        struct MotionBuffer * previousSolution,
                        //---------------------------------
                        struct BVH_Transform * bvhTargetTransform
                       )
{
    //Extrapolated guess
    struct MotionBuffer * extrapolatedGuess = mallocNewMotionBufferAndCopy(mc,solution);
    if (extrapolatedGuess!=0)
    {
        extrapolateSolution(
                            penultimateSolution,
                            previousSolution,
                            extrapolatedGuess
                           );

        ensureFinalProposedSolutionIsBetterInParts(
                                                   mc,
                                                   renderer,
                                                   //---------------------------------
                                                   "Extrapolation",
                                                   //---------------------------------
                                                   problem,
                                                   0,                       //2, //Start Chain
                                                   problem->numberOfChains-1, //problem->numberOfChains-1, //End Chain
                                                   //---------------------------------
                                                   ikConfig,
                                                   //---------------------------------
                                                   problem->currentSolution,
                                                   extrapolatedGuess,
                                                   //---------------------------------
                                                   bvhTargetTransform
                                                   //---------------------------------
                                                   );

        freeMotionBuffer(&extrapolatedGuess);
        return 1;
    }
  return 0;
}


int remapMotionBufferValues(struct BVH_MotionCapture * mc,struct MotionBuffer * buffer)
{
 if ( (buffer!=0) && (buffer->motion!=0) && (buffer->bufferSize==mc->numberOfValuesPerFrame) )
 {
  BVHMotionChannelID mID = 0;
  for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
   {
      BVHJointID jID = mc->motionToJointLookup[mID].jointID;
      if (
           (!mc->jointHierarchy[jID].hasQuaternionRotation) &&
           (!mc->jointHierarchy[jID].hasRodriguesRotation)  &&
            (mc->jointHierarchy[jID].hasRotationalChannels)
         )
      {
       if ( mc->motionToJointLookup[mID].channelID == BVH_ROTATION_X)
         { buffer->motion[mID] = bvh_normalizeAngle(buffer->motion[mID]); } else
       if ( mc->motionToJointLookup[mID].channelID == BVH_ROTATION_Y)
         { buffer->motion[mID] = bvh_normalizeAngle(buffer->motion[mID]); } else
       if ( mc->motionToJointLookup[mID].channelID == BVH_ROTATION_Z)
         { buffer->motion[mID] = bvh_normalizeAngle(buffer->motion[mID]); }
      }
   }
   return 1;
 }
 return 0;
}


int diagnoseMissing2DJoints(
                            struct BVH_MotionCapture * mc,
                            struct ikProblem * problem,
                            struct BVH_Transform * bvhTargetTransform
                           )
{
    int missing = 0;
    for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
    {
      for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
        {
          BVHJointID jID = problem->chain[chainID].part[partID].jID;
          float tX=(float) bvhTargetTransform->joint[jID].pos2D[0];
          float tY=(float) bvhTargetTransform->joint[jID].pos2D[1];
          if ( (tX==0.0) && (tY==0.0) )
          {
             fprintf(stderr,RED "Problem (%s) : IK Joint %u(%s) is missing \n" NORMAL,problem->problemDescription,jID,mc->jointHierarchy[jID].jointName);
             missing+=1;
          }
        }
    }

    return missing;
}




void enforceLimitsDirectlyOnMotionBuffer(
                                         struct BVH_MotionCapture * mc,
                                         struct ikProblem * problem,
                                         struct MotionBuffer * solution
                                        )
{
  //fprintf(stderr,MAGENTA " enforceLimitsDirectlyOnMotionBuffer on problem %s with %u chains \n" NORMAL,problem->problemDescription,problem->numberOfChains);
  for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
        {
          //fprintf(stderr,MAGENTA " Chain %u has %u parts\n" NORMAL,chainID,problem->chain[chainID].numberOfParts);
          for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
               {
                BVHJointID jID = problem->chain[chainID].part[partID].jID;
                char limitsEngaged = problem->chain[chainID].part[partID].limits;
                //fprintf(stderr,MAGENTA " Chain %u / Part %u / Joint %s / jID=%u / limits=%u\n" NORMAL,chainID,partID,mc->jointHierarchy[jID].jointName,jID,limitsEngaged);
                if (!problem->chain[chainID].part[partID].endEffector)
                {

                 if (limitsEngaged)
                 {
                  //--------------------------------------------------
                  unsigned int numberOfMIDElements = 1 + problem->chain[chainID].part[partID].mIDEnd - problem->chain[chainID].part[partID].mIDStart;
                  if (numberOfMIDElements!=3)
                  {
                    fprintf(stderr,RED "iteratePartLoss: %s Only 3 elements acceptable( got %u @ chain %u / part %u ) ..\n" NORMAL,problem->problemDescription,numberOfMIDElements,chainID,partID);
                    fprintf(stderr,RED "Joint %s / jID=%u\n" NORMAL,mc->jointHierarchy[jID].jointName,jID);
                    fprintf(stderr,RED "mIDStart: %u\n" NORMAL,problem->chain[chainID].part[partID].mIDStart);
                    fprintf(stderr,RED "mIDEnd: %u\n" NORMAL,problem->chain[chainID].part[partID].mIDEnd);
                    fprintf(stderr,RED "forcing 3 elements from %u -> %u\n" NORMAL,problem->chain[chainID].part[partID].mIDStart,problem->chain[chainID].part[partID].mIDStart+2);
                  }
                  //--------------------------------------------------
                  unsigned int mIDS[3] =
                  {
                     problem->chain[chainID].part[partID].mIDStart,
                     problem->chain[chainID].part[partID].mIDStart+1,
                     problem->chain[chainID].part[partID].mIDStart+2
                  };
                  //--------------------------------------------------

                    //---------------------------------------------------------------------------------------
                    float minimumLimitValues[3] = { problem->chain[chainID].part[partID].minimumLimitMID[0],
                                                    problem->chain[chainID].part[partID].minimumLimitMID[1],
                                                    problem->chain[chainID].part[partID].minimumLimitMID[2] };
                    float maximumLimitValues[3] = { problem->chain[chainID].part[partID].maximumLimitMID[0],
                                                    problem->chain[chainID].part[partID].maximumLimitMID[1],
                                                    problem->chain[chainID].part[partID].maximumLimitMID[2] };
                    //---------------------------------------------------------------------------------------
                    //fprintf(stderr,"limits engaged for Joint %s ",mc->jointHierarchy[jID].jointName);
                    //fprintf(stderr," %f/%f/%f => ",solution->motion[mIDS[0]],solution->motion[mIDS[1]],solution->motion[mIDS[2]]);
                    //fprintf(stderr," [ min %f/%f/%f -> max %f/%f/%f ]",minimumLimitValues[0],minimumLimitValues[1],minimumLimitValues[2],maximumLimitValues[0],maximumLimitValues[1],maximumLimitValues[2]);
                    char trigger = 0;
                    //---------------------------------------------------------------------------------------
                    if (solution->motion[mIDS[0]]<minimumLimitValues[0])  { solution->motion[mIDS[0]]=minimumLimitValues[0]; trigger=1; } else
                    if (solution->motion[mIDS[0]]>maximumLimitValues[0])  { solution->motion[mIDS[0]]=maximumLimitValues[0]; trigger=1; }
                    //-------------------------------------------------------------------------------------
                    if (solution->motion[mIDS[1]]<minimumLimitValues[1])  { solution->motion[mIDS[1]]=minimumLimitValues[1]; trigger=1; } else
                    if (solution->motion[mIDS[1]]>maximumLimitValues[1])  { solution->motion[mIDS[1]]=maximumLimitValues[1]; trigger=1; }
                    //-------------------------------------------------------------------------------------
                    if (solution->motion[mIDS[2]]<minimumLimitValues[2])  { solution->motion[mIDS[2]]=minimumLimitValues[2]; trigger=1; } else
                    if (solution->motion[mIDS[2]]>maximumLimitValues[2])  { solution->motion[mIDS[2]]=maximumLimitValues[2]; trigger=1; }
                    //-------------------------------------------------------------------------------------
                    //if (trigger) { fprintf(stderr,RED " "); }
                    //fprintf(stderr," = %f/%f/%f \n" NORMAL,solution->motion[mIDS[0]],solution->motion[mIDS[1]],solution->motion[mIDS[2]]);
                    //-------------------------------------------------------------------------------------
                 } //Joint has a limit thus it makes sense to test it
                } //not end effector
               } //loop over all parts
        } //loop over all chains
}












int approximateBodyFromMotionBufferUsingInverseKinematics(
                                                          struct BVH_MotionCapture * mc,
                                                          struct simpleRenderer *renderer,
                                                          struct ikProblem * problem,
                                                          struct ikConfiguration * ikConfig,
                                                          //---------------------------------
                                                          struct MotionBuffer * penultimateSolution,
                                                          struct MotionBuffer * previousSolution,
                                                          struct MotionBuffer * solution,
                                                          struct MotionBuffer * groundTruth,
                                                          //---------------------------------
                                                          struct BVH_Transform * bvhTargetTransform,
                                                          //---------------------------------
                                                          unsigned int useMultipleThreads,
                                                          //---------------------------------
                                                          float * initialMAEInPixels,
                                                          float * finalMAEInPixels,
                                                          float * initialMAEInMM,
                                                          float * finalMAEInMM
                                                         )
{
    if (problem==0)
    {
        fprintf(stderr,RED "No problem provided for IK..\n" NORMAL);
        return 0;
    }

    if (bvhTargetTransform==0)
    {
        fprintf(stderr,RED "No target transform, can't do IK..\n" NORMAL);
        return 0;
    }

    if  ( (solution==0) || (solution->motion==0) )
    {
        fprintf(stderr,RED "No initial solution provided for IK..\n" NORMAL);
        return 0;
    }

    if (ikConfig==0)
    {
        fprintf(stderr,RED "No configuration provided for IK..\n" NORMAL);
        return 0;
    }

    if (ikConfig->ikVersion != (float) IK_VERSION)
    {
        fprintf(stderr,RED "Fatal: IK Version mismatch for configuration structure (%0.2f vs %0.2f ) ..\n" NORMAL,ikConfig->ikVersion,IK_VERSION);
        fprintf(stderr,RED "There is something wrong with your setup, halting execution..\n" NORMAL);
        exit(0);
    }

    //Remap all motion buffers
    remapMotionBufferValues(mc,penultimateSolution);
    remapMotionBufferValues(mc,previousSolution);
    remapMotionBufferValues(mc,solution);
    remapMotionBufferValues(mc,groundTruth);

    //Make sure renderer gets its MV matrix calculated.. (since this is no longer done automatically)
    simpleRendererUpdateMovelViewTransform(renderer);

    unsigned long startTime = GetTickCountMicrosecondsIK();

    //Don't spam console..
    //viewProblem(problem);
    if (problem->chain[0].part[0].jID==0)
    {
      if (!ikConfig->dontUseSolutionHistory)
       {
         ensureInitialPositionIsInFrustrum(problem,renderer,solution,previousSolution);
       }
    } else
    if (ikConfig->verbose)
    {
      fprintf(stderr,RED "%s problem, Not running initial position frustrum check\n" NORMAL,problem->problemDescription);
    }

    //Make sure our problem has the correct details ..
    problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

    if (diagnoseMissing2DJoints(mc,problem,bvhTargetTransform))
    {
      fprintf(stderr,RED "IK has missing target joints\n" NORMAL);
    }

    if (!updateProblemSolutionToAllChains(problem,solution))
           {
             fprintf(stderr,RED "Failed broadcasting current solution to all chains\n" NORMAL);
             return 0;
           }
    if (!copyMotionBuffer(problem->previousSolution,previousSolution) )
           {
             fprintf(stderr,RED "Failed copying previous solution to problem\n" NORMAL);
             return 0;
           }

    //Extrapolated guess
    doExtrapolatedGuess(
                        mc,
                        renderer,
                        problem,
                        ikConfig,
                        //---------------------------------
                        solution,
                        penultimateSolution,
                        previousSolution,
                        //---------------------------------
                        bvhTargetTransform
                       );

    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    struct BVH_Transform bvhCurrentTransform= {0};

    if (bvh_loadTransformForMotionBuffer(mc,problem->initialSolution->motion,&bvhCurrentTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
    {
        //----------------------------------------------------
        if (initialMAEInPixels!=0)
        {
            *initialMAEInPixels = meanBVH2DDistance(mc,renderer,1,0,&bvhCurrentTransform,bvhTargetTransform,ikConfig->verbose);
        }
        //----------------------------------------------------
        if ( (initialMAEInMM!=0) && (groundTruth!=0) )
        {
            *initialMAEInMM = meanBVH3DDistance(mc,renderer,1,0,problem->initialSolution->motion,&bvhCurrentTransform,groundTruth->motion,bvhTargetTransform);
        }
        //----------------------------------------------------
    }

    if (ikConfig->dumpScreenshots)
    {
        dumpBVHToSVGFrame("initial.svg",mc,&bvhCurrentTransform,0,renderer);
    }
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
     if (useMultipleThreads)
     {
      //Solve the problem using multiple threads..!
      //Due to game theory and the lack of information sharing between different "Actors"
      //The accuracy of this is worse than the single threaded version
      multiThreadedSolver(problem,ikConfig);
     } else
     {
     //Solve the problem using a single thread..!
      singleThreadedSolver(problem,ikConfig);
     }

    //Retrieve regressed solution
    copyMotionBuffer(solution,problem->currentSolution);

    if (ikConfig->verbose)
     {
      float * m = problem->initialSolution->motion;
      if (m!=0)
      {
       fprintf(stderr,"Initial Position/Location was %0.2f,%0.2f,%0.2f %0.2f,%0.2f,%0.2f\n",m[0],m[1],m[2],m[3],m[4],m[5]);

        if (!ikConfig->dontUseSolutionHistory)
        {
         if  ( (problem->previousSolution!=0) && (problem->previousSolution->motion!=0) )
         {
            m = problem->previousSolution->motion;
            fprintf(stderr,"Previous Position/Location was %0.2f,%0.2f,%0.2f %0.2f,%0.2f,%0.2f\n",m[0],m[1],m[2],m[3],m[4],m[5]);
         }
        }

       m = solution->motion;
       fprintf(stderr,"Final Position/Location was %0.2f,%0.2f,%0.2f %0.2f,%0.2f,%0.2f\n",m[0],m[1],m[2],m[3],m[4],m[5]);
      }
     }
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    bvh_markAllJointsAsUsefullInTransform(mc,&bvhCurrentTransform);

    if (
          bvh_loadTransformForMotionBuffer(
                                           mc,
                                           solution->motion,
                                           &bvhCurrentTransform,
                                           ikConfig->penalizeSymmetriesHeuristic// dont use extra structures
                                          )
      )
    {
        //Is the pointer there ? if not it means we don't care about final MAE
        if (finalMAEInPixels!=0)
        {
        //We calculate the new 2D distance achieved..
        *finalMAEInPixels  = meanBVH2DDistance(
                                                mc,
                                                renderer,
                                                1, //useAllJoints
                                                0, //onlyConsiderChildrenOfThisJoint
                                                &bvhCurrentTransform,
                                                bvhTargetTransform,
                                                ikConfig->verbose
                                               );

        //Was our solution perfect? If it was we don't need to compare to previous
        //----------------------------------------------------
        if (*finalMAEInPixels!=0) //there is no perfect solution!
        {
           if (!ikConfig->dontUseSolutionHistory)
           {
           //Perform projection on previous solution
           //-----------------------------------------------
           struct BVH_Transform bvhPreviousTransform = {0};
           if (bvh_loadTransformForMotionBuffer(mc,problem->previousSolution->motion,&bvhPreviousTransform,ikConfig->penalizeSymmetriesHeuristic))// We don't need extra structures
           {
            float previousMAEInPixels =  meanBVH2DDistance(mc,renderer,1,0,&bvhPreviousTransform,bvhTargetTransform,ikConfig->verbose);

            if (previousMAEInPixels<*finalMAEInPixels)
              {
                if (ikConfig->considerPreviousSolution)
                {
                    fprintf(stderr,RED "After all this work we where not smart enough to understand that previous solution was better all along..\n" NORMAL);
                    copyMotionBuffer(solution,previousSolution);
                }
              }
            }
            bvh_freeTransform(&bvhPreviousTransform);
           //-----------------------------------------------
           }
        } else
        {
          fprintf(stderr,GREEN "We have a perfect solution(?) Too good to be true..\n" NORMAL);
        }
       }
        //----------------------------------------------------
        if ( (finalMAEInMM!=0) && (groundTruth!=0) )
        {
            *finalMAEInMM = meanBVH3DDistance(mc,renderer,1,0,solution->motion,&bvhCurrentTransform,groundTruth->motion,bvhTargetTransform);
        }
        //----------------------------------------------------
        /*
        //This is smoother but worse..
        springToZeroParts(
                           mc,
                           renderer,
                           //---------------------------------
                           problem,
                           2, //Start Chain
                           problem->numberOfChains, //End Chain
                           //---------------------------------
                           ikConfig,
                           //---------------------------------
                           solution,
                           //---------------------------------
                           bvhTargetTransform
                           //---------------------------------
                          );
        */



        if ( (!ikConfig->dontUseSolutionHistory) && (previousSolution!=0) && (previousSolution->motion!=0) && (previousSolution->bufferSize==solution->bufferSize) )
        {
         //This removes some weird noise from previous solution
         ensureFinalProposedSolutionIsBetterInParts(
                                                    mc,
                                                    renderer,
                                                    "Previous",
                                                    problem,
                                                    0,                       //2, //Start Chain
                                                    problem->numberOfChains-1, //problem->numberOfChains-1, //End Chain
                                                    ikConfig,
                                                    solution,
                                                    previousSolution,
                                                    bvhTargetTransform
                                                   );
        }

        if ( (!ikConfig->dontUseSolutionHistory) && (penultimateSolution!=0) && (penultimateSolution->motion!=0) && (penultimateSolution->bufferSize==solution->bufferSize) )
        {
         //This removes some weird noise from pre-previous solution
         ensureFinalProposedSolutionIsBetterInParts(
                                                    mc,
                                                    renderer,
                                                    "Penultimate",
                                                    problem,
                                                    0,                       //2, //Start Chain
                                                    problem->numberOfChains-1, //problem->numberOfChains-1, //End Chain
                                                    ikConfig,
                                                    solution,
                                                    penultimateSolution,
                                                    bvhTargetTransform
                                                   );
        }


    } else
    {
      fprintf(stderr,RED "Failed loading solution transform for problem\n" NORMAL);
    }
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------

    if (ikConfig->dumpScreenshots)
    {
        dumpBVHToSVGFrame("target.svg",mc,bvhTargetTransform,1,renderer);
        dumpBVHToSVGFrame("solution.svg",mc,&bvhCurrentTransform,0,renderer);
    }

    //---------------------------------------------
    enforceLimitsDirectlyOnMotionBuffer(
                                         mc,
                                         problem,
                                         solution
                                        );
    //---------------------------------------------

    unsigned long endTime = GetTickCountMicrosecondsIK();

     if (useMultipleThreads)
        { fprintf(stderr,RED "MT" NORMAL); }

    if (ikConfig->useLangevinDynamics)
        { fprintf(stderr,RED "L" NORMAL); }

    fprintf(stderr,"IK %lu μsec|%s|lr=%0.3f|b.loss=%0.1f|it.s=%u|epochs=%u\n",
                                               endTime-startTime,
                                               problem->problemDescription,
                                               ikConfig->learningRate,
                                               problem->chain[0].currentLoss,
                                               ikConfig->iterations,
                                               ikConfig->epochs
                                               );
    bvh_freeTransform(&bvhCurrentTransform);

    return 1;
}


