/** @file main.c
 *  @brief  A library that can parse BVH files and perform various processing options as a commandline tool
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../../Library/TrajectoryParser/TrajectoryParserDataStructures.h"
#include "../../Library/MotionCaptureLoader/bvh_loader.h"
#include "../../Library/MotionCaptureLoader/calculate/bvh_to_tri_pose.h"
#include "../../Library/MotionCaptureLoader/calculate/smoothing.h"

#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.h"
#include "../../Library/MotionCaptureLoader/export/bvh_export.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_bvh.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_csv.h"
#include "../../Library/MotionCaptureLoader/export/bvh_to_c.h"

#include "../../Library/MotionCaptureLoader/edit/bvh_cut_paste.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_randomize.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_filter.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_rename.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_merge.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_remapangles.h"
#include "../../Library/MotionCaptureLoader/edit/bvh_interpolate.h"

#include "../../Library/MotionCaptureLoader/ik/bvh_inverseKinematics.h"
#include "../../Library/MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.h"

#include "../../Library/MotionCaptureLoader/metrics/bvh_measure.h"
#include "../../Library/MotionCaptureLoader/tests/test.h"

#include  "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include  "../../../../../tools/AmMatrix/matrixOpenGL.h"

// Opaque handle type — callers only ever see void*
typedef void * BVHHandle;

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

void haltOnError(unsigned int haltingSwitch,const char * message)
{
  fprintf(stderr,RED "=======================================\n");
  fprintf(stderr,"=======================================\n");
  fprintf(stderr,"Encountered error during procedure %s \n",message);
  fprintf(stderr,"=======================================\n");
  fprintf(stderr,"=======================================\n" NORMAL);

  if (haltingSwitch)
    {
       fprintf(stderr,RED "Halting because of --haltonerror switch\n" NORMAL);
       exit(1);
    }
}

void incorrectArguments()
{
  fprintf(stderr,RED "Incorrect number of arguments.. \n" NORMAL);
  exit(1);
}

//------------------------------------------------------------------
//------------------------------------------------------------------
//------------------------------------------------------------------
//------------------------------------------------------------------

// All BVH state for one loaded file lives in this struct.
// It is heap-allocated by bvhConverter_loadAtomic / bvh_createContext
// and freed by bvhConverter_unloadAtomic / bvh_destroyContext.
struct BVHContext
{
    struct BVH_MotionCapture         motion;
    struct BVH_Transform             transform;
    struct simpleRenderer            renderer;
    struct BVH_RendererConfiguration rendererConfig;

    struct ikProblem * face;
    struct ikProblem * body;
    struct ikProblem * lhand;
    struct ikProblem * rhand;

    struct MotionBuffer * penultimate;
    struct MotionBuffer * previous;
    struct MotionBuffer * solution;

    struct ButterWorthArray * filter;
};

//------------------------------------------------------------------
//------------------------------------------------------------------
//------------------------------------------------------------------
//------------------------------------------------------------------

// Convenience macro: cast the opaque handle at the top of every function.
#define CTX(h) ((struct BVHContext*)(h))


BVHHandle bvhConverter_loadAtomic(const char *path)
{
  struct BVHContext *ctx = (struct BVHContext*) calloc(1, sizeof(struct BVHContext));
  if (ctx == NULL)
  {
      fprintf(stderr,RED "bvhConverter_loadAtomic: failed to allocate BVHContext\n" NORMAL);
      return NULL;
  }

  float scaleWorld=1.0;
  int immediatelyHaltOnError = 1;
  fprintf(stderr,"Attempting to load %s\n",path);
  if (!bvh_loadBVH(path, &ctx->motion, scaleWorld))
          {
            haltOnError(immediatelyHaltOnError,"Error loading bvh file..");
          }
  //Change joint names..
  bvh_renameJointsForCompatibility(&ctx->motion);


  // Emulate GoPro Hero4 @ FullHD mode by default..
  // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
  ctx->rendererConfig.near   = 1.0;
  ctx->rendererConfig.far    = 10000.0;
  ctx->rendererConfig.width  = 1920;
  ctx->rendererConfig.height = 1080;
  ctx->rendererConfig.cX     = (float)ctx->rendererConfig.width/2;
  ctx->rendererConfig.cY     = (float)ctx->rendererConfig.height/2;
  ctx->rendererConfig.fX     = 582.18394;
  ctx->rendererConfig.fY     = 582.52915;
  //----------------------------------------------
  simpleRendererDefaults(
                          &ctx->renderer,
                          ctx->rendererConfig.width,
                          ctx->rendererConfig.height,
                          ctx->rendererConfig.fX,
                          ctx->rendererConfig.fY
                         );
  //----------------------------------------------
  simpleRendererInitialize(&ctx->renderer);
  //----------------------------------------------
  return (BVHHandle) ctx;
}


int bvhConverter_unloadAtomic(BVHHandle ctxHandle)
{
    if (ctxHandle == NULL) { return 1; }
    struct BVHContext *ctx = CTX(ctxHandle);

    // Free heap-allocated sub-members (IK problems, motion buffers, smoothing filter).
    // TODO: add dedicated free functions for BVH_MotionCapture and BVH_Transform internals.
    if (ctx->face)        { free(ctx->face);        ctx->face        = NULL; }
    if (ctx->body)        { free(ctx->body);         ctx->body        = NULL; }
    if (ctx->lhand)       { free(ctx->lhand);        ctx->lhand       = NULL; }
    if (ctx->rhand)       { free(ctx->rhand);        ctx->rhand       = NULL; }
    if (ctx->penultimate) { free(ctx->penultimate);  ctx->penultimate = NULL; }
    if (ctx->previous)    { free(ctx->previous);     ctx->previous    = NULL; }
    if (ctx->solution)    { free(ctx->solution);     ctx->solution    = NULL; }
    if (ctx->filter)      { free(ctx->filter);       ctx->filter      = NULL; }

    free(ctx);
    return 1;
}

int bvhConverter_rendererConfigurationAtomic(BVHHandle ctxHandle,const char ** labels,const float * values,int numberOfElements)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  // Emulate GoPro Hero4 @ FullHD mode by default..
  // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
  ctx->rendererConfig.near   = 1.0;
  ctx->rendererConfig.far    = 10000.0;
  ctx->rendererConfig.width  = 1920;
  ctx->rendererConfig.height = 1080;
  ctx->rendererConfig.cX     = (float)ctx->rendererConfig.width/2;
  ctx->rendererConfig.cY     = (float)ctx->rendererConfig.height/2;
  ctx->rendererConfig.fX     = 582.18394;
  ctx->rendererConfig.fY     = 582.52915;

  fprintf(stderr,"bvhConverter_rendererConfigurationAtomic received %u elements\n",numberOfElements);
  for (int i=0; i<numberOfElements; i++)
  {
      //fprintf(stderr," %u - %s->%0.2f\n",i,labels[i],values[i]);
      if (strcmp(labels[i],"near")==0)   { ctx->rendererConfig.near   = values[i]; } else
      if (strcmp(labels[i],"far")==0)    { ctx->rendererConfig.far    = values[i]; } else
      if (strcmp(labels[i],"width")==0)  { ctx->rendererConfig.width  = (unsigned int) values[i]; } else
      if (strcmp(labels[i],"height")==0) { ctx->rendererConfig.height = (unsigned int) values[i]; } else
      if (strcmp(labels[i],"cX")==0)     { ctx->rendererConfig.cX     = values[i]; } else
      if (strcmp(labels[i],"cY")==0)     { ctx->rendererConfig.cY     = values[i]; } else
      if (strcmp(labels[i],"fX")==0)     { ctx->rendererConfig.fX     = values[i]; } else
      if (strcmp(labels[i],"fY")==0)     { ctx->rendererConfig.fY     = values[i]; } else
        {
          fprintf(stderr,RED"bvhConverter_rendererConfigurationAtomic: Unknown command %u - %s->%0.2f\n" NORMAL,i,labels[i],values[i]);
        }
  }

   simpleRendererDefaults(
                          &ctx->renderer,
                          ctx->rendererConfig.width,
                          ctx->rendererConfig.height,
                          ctx->rendererConfig.fX,
                          ctx->rendererConfig.fY
                         );
    simpleRendererInitialize(&ctx->renderer);
  return 1;
}

int bvhConverter_processFrame(BVHHandle ctxHandle,int frameID)
{
    struct BVHContext *ctx = CTX(ctxHandle);
    int occlusions=1;
    return performPointProjectionsForFrame(
                                           &ctx->motion,
                                           &ctx->transform,
                                           frameID,
                                           &ctx->renderer,
                                           occlusions,
                                           ctx->rendererConfig.isDefined
                                          );
}


int bvhConverter_scale(BVHHandle ctxHandle,float scaleRatio)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  fprintf(stderr,"Offset scaling ratio = %0.2f \n",scaleRatio);
  return bvh_scaleAllOffsets(
                              &ctx->motion,
                              scaleRatio
                            );
}

int bvhConverter_getNumberOfMotionValuesPerFrame(BVHHandle ctxHandle)
{
 return CTX(ctxHandle)->motion.numberOfValuesPerFrame;
}

int bvhConverter_getNumberOfJoints(BVHHandle ctxHandle)
{
 return CTX(ctxHandle)->motion.jointHierarchySize;
}

int bvhConverter_writeBVH(BVHHandle ctxHandle,char * filename,int writeHierarchy,int writeMotion)
{
 return dumpBVHToBVH(
                     filename,
                     &CTX(ctxHandle)->motion,
                     writeHierarchy,
                     writeMotion
                    );
}

int bvhConverter_getMotionValueOfFrame(BVHHandle ctxHandle,int fID,int mID)
{
 return bvh_getMotionValueOfFrame(&CTX(ctxHandle)->motion,fID,mID);
}

int bvhConverter_setMotionValueOfFrame(BVHHandle ctxHandle,int fID,int mID,float value)
{
 float localValue = value;
 return bvh_setMotionValueOfFrame(&CTX(ctxHandle)->motion,fID,mID,&localValue);
}

int bvhConverter_getJointNameJointID(BVHHandle ctxHandle,const char * jointName)
{
  //fprintf(stderr,"Asked to resolve %s\n",jointName);
  BVHJointID jID=0;
  if  (
        bvh_getJointIDFromJointNameNocase(
                                          &CTX(ctxHandle)->motion,
                                          jointName,
                                          &jID
                                         )
      )
      {
       return jID;
      }
  fprintf(stderr,RED "BVH library could not resolve joint \"%s\" \n" NORMAL,jointName);
  return -1;
}

const char * bvhConverter_getJointNameFromJointID(BVHHandle ctxHandle,int jointID)
{
    struct BVHContext *ctx = CTX(ctxHandle);
    if (jointID<ctx->motion.jointHierarchySize)
    {
        return ctx->motion.jointHierarchy[jointID].jointName;
    }
  fprintf(stderr,RED "BVH library could not resolve joint name for joint out bounds \"%u\" \n" NORMAL,jointID);
  return "";
}

int bvhConverter_getJointParent(BVHHandle ctxHandle,int jointID)
{
    struct BVHContext *ctx = CTX(ctxHandle);
    if (jointID<ctx->motion.jointHierarchySize)
    {
        return ctx->motion.jointHierarchy[jointID].parentJoint;
    }
  fprintf(stderr,RED "BVH library could not resolve joint parent for joint out bounds \"%u\" \n" NORMAL,jointID);
  return 0;
}

float bvhConverter_get3DX(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get3DX(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return ctx->transform.joint[jointID].pos3D[0]; }
  return 0.0;
}

float  bvhConverter_get3DY(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get3DY(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return ctx->transform.joint[jointID].pos3D[1]; }
  return 0.0;
}

float  bvhConverter_get3DZ(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get3DZ(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return ctx->transform.joint[jointID].pos3D[2]; }
  return 0.0;
}

float bvhConverter_get2DX(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get2DX(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return (float) ctx->transform.joint[jointID].pos2D[0]/ctx->rendererConfig.width; }
  return 0.0;
}

float  bvhConverter_get2DY(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get2DY(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return (float) ctx->transform.joint[jointID].pos2D[1]/ctx->rendererConfig.height; }
  return 0.0;
}



int bvhConverter_isJointEndSite(BVHHandle ctxHandle,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  if (jointID<ctx->motion.jointHierarchySize)
     { return ctx->motion.jointHierarchy[jointID].isEndSite; }
  fprintf(stderr,RED "BVH library could not resolve if joint is an EndSite because joint is out bounds \"%u\" \n" NORMAL,jointID);
  return 0;
}

float bvhConverter_getBVHJointRotationXForFrame(BVHHandle ctxHandle,int frameID,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get2DX(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return (float) bvh_getJointRotationXAtFrame(&ctx->motion,jointID,frameID); }
  return 0.0;
}
float bvhConverter_getBVHJointRotationYForFrame(BVHHandle ctxHandle,int frameID,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get2DX(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return (float) bvh_getJointRotationYAtFrame(&ctx->motion,jointID,frameID); }
  return 0.0;
}
float bvhConverter_getBVHJointRotationZForFrame(BVHHandle ctxHandle,int frameID,int jointID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_get2DX(%u)\n",jointID);
  if (jointID<ctx->transform.numberOfJointsToTransform)
     { return (float) bvh_getJointRotationZAtFrame(&ctx->motion,jointID,frameID); }
  return 0.0;
}



int bvhConverter_modifySingleAtomic(BVHHandle ctxHandle,const char * label,const float value,int frameID)
{
  struct BVHContext *ctx = CTX(ctxHandle);
  if (label==0) { return 0; }
  if (frameID>=ctx->motion.numberOfFrames) { return 0; }
  //------------------------------------------------------------
  //fprintf(stderr,"bvhConverter_modifyAtomic received element %s with value %0.2f for frame %u\n",label,value,frameID);
  //------------------------------------------------------------
  int everythingOk = 1;
  char jointName[513]={0};
  snprintf(jointName,512,"%s",label);
  char * delimeter = strchr(jointName,'_');
  if (delimeter==0)
  {
      fprintf(stderr,"bvhConverter_modifyAtomic received element %s with value %0.2f for frame %u ",label,value,frameID);
      fprintf(stderr,"it doesn't have a degree of freedom associated so ignoring it..");
      return 0;
  }
  *delimeter = 0;
  char * dof = delimeter+1;
  //=======================================================
  lowercase(jointName);
  lowercase(dof);
  //=======================================================
  if (strstr(jointName,"endsite_")!=0)
  {
     fprintf(stderr,RED "Endsites can't be modified..!\n" NORMAL);
     return 1;
  }
  if (strstr(jointName,"padding")!=0)
  {
     fprintf(stderr,RED "Paddings can't be modified..!\n" NORMAL);
     return 1;
  }



  if (strcmp(jointName,"neck01")==0)
  {
    snprintf(jointName,512,"neck1"); //Fix ?
  }
  if (strcmp(jointName,"lthumbbase")==0)
  {
    snprintf(jointName,512,"__lthumb"); //Fix ?
  }
  if (strcmp(jointName,"rthumbbase")==0)
  {
    snprintf(jointName,512,"__rthumb"); //Fix ?
  }

  //fprintf(stderr," %s->%0.2f ",label,value);
  //fprintf(stderr," Joint:%s Control:%s\n",jointName,dof);
  //=======================================================
  //int jointID = bvhConverter_getJointNameJointID(ctxHandle,jointName);
  BVHJointID jointID=0;
  if  (
        bvh_getJointIDFromJointNameNocase(
                                          &ctx->motion,
                                          jointName,
                                          &jointID
                                         )
      )
      {
      // The next line is a debug message that spams a *lot*!
      //fprintf(stderr,"Joint ID %u / %s|%s => %0.2f \n",jointID,ctx->motion.jointHierarchy[jointID].jointName,dof,value);
      //==============================================================================================================
      if (strcmp(dof,"xposition")==0) { bvh_setJointPositionXAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"yposition")==0) { bvh_setJointPositionYAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"zposition")==0) { bvh_setJointPositionZAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"xrotation")==0) { bvh_setJointRotationXAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"yrotation")==0) { bvh_setJointRotationYAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"zrotation")==0) { bvh_setJointRotationZAtFrame(&ctx->motion,jointID,frameID,value); } else
      if (strcmp(dof,"wrotation")==0) { bvh_setJointRotationWAtFrame(&ctx->motion,jointID,frameID,value); } else
                                      {
                                         fprintf(stderr,RED "\n\n\nBVH library could not perform modification  \"%s\" for joint \"%s\" \n\n\n" NORMAL,dof,jointName);
                                         everythingOk=0;
                                      }
      //==============================================================================================================
      } else
      {
          fprintf(stderr,RED "\nBVH library modification could not resolve joint \"%s\" \n" NORMAL,jointName);
          everythingOk=0;
      }
  return everythingOk;
}



int bvhConverter_modifyAtomic(BVHHandle ctxHandle,const char ** labels,const float * values,int numberOfElements,int frameID)
{
  //fprintf(stderr,"bvhConverter_modifyAtomic received %u elements\n",numberOfElements);
  int everythingOk = 1;
  for (int i=0; i<numberOfElements; i++)
  {
      if (!bvhConverter_modifySingleAtomic(ctxHandle,labels[i],values[i],frameID))
      {
          fprintf(stderr,RED "\n\n\nBVH library modification failed resolving all joints\n\n\n" NORMAL);
          everythingOk=0;
      }
  }
  return everythingOk;
}


int bvhConverter_eraseHistory(BVHHandle ctxHandle,int frameID)
{
    struct BVHContext *ctx = CTX(ctxHandle);
    int i=0;
    if (ctx->penultimate!=0)
    {
        for (i=0; i<ctx->penultimate->bufferSize; i++)
           { ctx->penultimate->motion[i]=0.0; }
    }

    if (ctx->previous!=0)
    {
        for (i=0; i<ctx->previous->bufferSize; i++)
           { ctx->previous->motion[i]=0.0; }
    }

    if (ctx->solution!=0)
    {
        for (i=0; i<ctx->solution->bufferSize; i++)
           { ctx->solution->motion[i]=0.0; }
    }
 return 1;
}


int bvhConverter_IKSetup(BVHHandle ctxHandle,const char * bodyPart,const char ** labels,const float * values,int numberOfElements,int frameID)
{
    struct BVHContext *ctx = CTX(ctxHandle);
    struct ikProblem * problem = 0;

    //Handle current/previous solution (assuming bvhConverter_modifyAtomic has been called)..
    //=======================================================================================
    if (ctx->penultimate==0)
          { ctx->penultimate = mallocNewMotionBuffer(&ctx->motion); }
    //=======================================================================================
    if (ctx->previous==0)
          { ctx->previous    = mallocNewMotionBuffer(&ctx->motion); }
     //====================================================================
    if (ctx->solution==0)
          { ctx->solution    = mallocNewMotionBuffer(&ctx->motion); }
     //====================================================================


    //Construct and/or select the correct problem to solve..!
    if (strcmp(bodyPart,"body")==0)
    {
       if (ctx->body==0)
           {
            fprintf(stderr,GREEN "Initializing Body Problem for the first time..\n" NORMAL);
            ctx->body = allocateEmptyIKProblem();
            prepareDefaultBodyProblem(
                                          ctx->body,
                                          &ctx->motion,
                                          &ctx->renderer,
                                          ctx->previous,
                                          ctx->solution,
                                          &ctx->transform
                                         );
            fprintf(stderr,GREEN "Done Initializing Body Problem ..\n" NORMAL);
           }
      problem = ctx->body;
    } else
    if (strcmp(bodyPart,"face")==0)
    {
      if (ctx->face==0)
           {
            fprintf(stderr,GREEN "Initializing Face Problem for the first time..\n" NORMAL);
              ctx->face = allocateEmptyIKProblem();
              prepareDefaultFaceProblem(
                                        ctx->face,
                                        &ctx->motion,
                                        &ctx->renderer,
                                        ctx->previous,
                                        ctx->solution,
                                        &ctx->transform,
                                        1
                                       );
            fprintf(stderr,GREEN "Done Initializing Face Problem ..\n" NORMAL);
           }
      problem = ctx->face;
    } else
    if (strcmp(bodyPart,"rhand")==0)
    {
       if (ctx->rhand==0)
           {
            fprintf(stderr,GREEN "Initializing RHand Problem for the first time..\n" NORMAL);
            ctx->rhand = allocateEmptyIKProblem();
            prepareDefaultRightHandProblem(
                                          ctx->rhand,
                                          &ctx->motion,
                                          &ctx->renderer,
                                          ctx->previous,
                                          ctx->solution,
                                          &ctx->transform,
                                          1
                                         );
            fprintf(stderr,GREEN "Done Initializing RHand Problem ..\n" NORMAL);
           }
       problem = ctx->rhand;
    } else
    if (strcmp(bodyPart,"lhand")==0)
    {
       if (ctx->lhand==0)
           {
            fprintf(stderr,GREEN "Initializing LHand Problem for the first time..\n" NORMAL);
            ctx->lhand = allocateEmptyIKProblem();
            prepareDefaultLeftHandProblem(
                                          ctx->lhand,
                                          &ctx->motion,
                                          &ctx->renderer,
                                          ctx->previous,
                                          ctx->solution,
                                          &ctx->transform,
                                          1
                                         );
            fprintf(stderr,GREEN "Done Initializing LHand Problem ..\n" NORMAL);
           }
       problem = ctx->lhand;
    }

    if (problem==0)
    {
      fprintf(stderr,"bvhConverter_IKSetup: Unrecognized body part `%s` \n",bodyPart);
      return 0;
    } else
    {
      return 1;
    }
}


float bvhConverter_IKFineTune(
                               BVHHandle ctxHandle,
                               const char * bodyPart,
                               const char ** labels,
                               const float * values,
                               int numberOfElements,
                               int frameID,
                               int iterations,
                               int epochs,
                               float lr,
                               float fSampling,
                               float fCutoff,
                               float langevinDynamics
                             )
{
  struct BVHContext *ctx = CTX(ctxHandle);
  //fprintf(stderr,"bvhConverter_IKFineTune(Part %s,Elements %u, Frame %u)\n",bodyPart,numberOfElements,frameID);

  //-----------------------------
  float initialMAEInPixels = 10000.0;
  float finalMAEInPixels   = 10000.0;
  float initialMAEInMM     = 10000.0;
  float finalMAEInMM       = 10000.0;
  //-----------------------------
  //By default select no problem..
  struct ikProblem * selectedProblem  = 0;
  //-----------------------------
  int initializeIK = 0;
  if (strcmp(bodyPart,"body")==0)
    {
       selectedProblem = ctx->body;
       if (ctx->body==0)   { initializeIK=1; }
    } else
  if (strcmp(bodyPart,"rhand")==0)
    {
       selectedProblem = ctx->rhand;
       if (ctx->rhand==0)  { initializeIK=1; }
    } else
  if (strcmp(bodyPart,"lhand")==0)
    {
       selectedProblem = ctx->lhand;
       if (ctx->lhand==0)  { initializeIK=1; }
    } else
  if (strcmp(bodyPart,"face")==0)
    {
       selectedProblem = 0;//Just run Butterworth, no face IK
       if (ctx->face==0)   { initializeIK=1; }
    }
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    if (initializeIK)
    {
     bvhConverter_IKSetup(ctxHandle,bodyPart,labels,values,numberOfElements,frameID);
    }

    if (ctx->filter==0)
    {
     fprintf(stderr,"Butterworth Smoothing filter initialized with fSampling:%0.2f and fCutoff:%0.2f \n",fSampling,fCutoff);
     ctx->filter = butterWorth_allocate(ctx->motion.numberOfValuesPerFrame,fSampling,fCutoff);
    }
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------

    //------------------------------------
         struct ikConfiguration ikConfig        = {0};
         ikConfig.learningRate                  = lr;
         ikConfig.iterations                    = iterations;
         ikConfig.epochs                        = epochs;
         ikConfig.maximumAcceptableStartingLoss = 50000;//12000; //WARING < -  consider setting this to 0
         ikConfig.gradientExplosionThreshold    = 25; //50 default
         ikConfig.iterationEarlyStopping        = 1;  //<-
         ikConfig.iterationMinimumLossDelta     = 10; //<- losses seem to be numbers 2000 -> 300 so 10 is a good limit
         ikConfig.spring                        = 20;
         ikConfig.dumpScreenshots               = 0; // Dont thrash disk
         ikConfig.verbose                       = 0; //Dont spam console
         ikConfig.tryMaintainingLocalOptima     = 1; //Less Jittery but can be stuck at local optima
         ikConfig.dontUseSolutionHistory        = 0;
         ikConfig.useLangevinDynamics           = langevinDynamics;
         ikConfig.ikVersion                     = IK_VERSION;
         //------------------------------------

        int multiThreading = 0;

         //======================================================================================================
         //======================================================================================================
         //======================================================================================================
        if ( (strcmp(bodyPart,"body")==0) || (strcmp(bodyPart,"lhand")==0) || (strcmp(bodyPart,"rhand")==0) || (strcmp(bodyPart,"face")==0) )
        {
         //Keep history..!
         copyMotionBuffer(ctx->penultimate,ctx->previous);
         copyMotionBuffer(ctx->previous,ctx->solution);
         bvh_copyMotionFrameToMotionBuffer(
                                           &ctx->motion,
                                           ctx->solution,
                                           frameID
                                          );

         char jointName[512]={0};
         struct BVH_Transform bvhTargetTransform={0};
         int occlusions=1;
         performPointProjectionsForFrame(
                                           &ctx->motion,
                                           &bvhTargetTransform,
                                           frameID,
                                           &ctx->renderer,
                                           occlusions,
                                           ctx->rendererConfig.isDefined
                                        );

         for (int i=0; i<numberOfElements; i++)
         {
             //fprintf(stderr,"Number %u => %s with %0.2f \n",i,labels[i],values[i] );
             snprintf(jointName,512,"%s",labels[i]);
             char * delimeter = strchr(jointName,'_');

             if (delimeter!=0)
             {
             *delimeter = 0;
             char * coord = jointName;
             char * dof   = delimeter+1;
             //=======================================================
             lowercase(coord);
             lowercase(dof);

             if (  (coord[0]=='2') && (coord[1]=='d') && (  (coord[2]=='x') || (coord[2]=='y') ) )
             {
              BVHJointID jID=0;
              if ( bvh_getJointIDFromJointNameNocase(&ctx->motion,dof,&jID) )
                {
                  if (coord[2]=='x')
                  {
                   //fprintf(stderr,GREEN "%s/%s \n" NORMAL,coord,dof);
                   bvhTargetTransform.joint[jID].pos2D[0] = (float) values[i]*ctx->rendererConfig.width;
                  } else
                  if (coord[2]=='y')
                  {
                   //fprintf(stderr,GREEN "%s/%s \n" NORMAL,coord,dof);
                   bvhTargetTransform.joint[jID].pos2D[1] = (float) values[i]*ctx->rendererConfig.height;
                  }
                } else
                {
                  //fprintf(stderr,RED "IK: Could not resolve Joint %s for Number %u => %s with %0.2f \n" NORMAL,dof,i,labels[i],values[i] );
                }
            }//2DX/Y

           } //Tag has an _ and we process it
         }//Loop over received elements

         if (  approximateBodyFromMotionBufferUsingInverseKinematics(
                                                                     &ctx->motion,
                                                                     &ctx->renderer,
                                                                     selectedProblem,
                                                                     &ikConfig,
                                                                     //----------------
                                                                     ctx->penultimate,
                                                                     ctx->previous,
                                                                     ctx->solution,
                                                                     0, //No ground truth..
                                                                     //----------------
                                                                     &bvhTargetTransform,
                                                                     //----------------
                                                                     multiThreading,// 0=single thread, 1=multi thread
                                                                     //----------------
                                                                     &initialMAEInPixels,
                                                                     &finalMAEInPixels,
                                                                     &initialMAEInMM,
                                                                     &finalMAEInMM
                                                                    )
            )
            {
              if(!bvh_copyMotionBufferToMotionFrame(
                                                    &ctx->motion,
                                                    frameID,
                                                    ctx->solution
                                                   )
                )
                {
                    fprintf(stderr,RED "Failed bvh_copyMotionBufferToMotionFrame\n" NORMAL);
                }

                //Perform and update projections for new results..!
                bvhConverter_processFrame(ctxHandle,frameID);
            } else
            {
              fprintf(stderr,RED "Failed approximateBodyFromMotionBufferUsingInverseKinematics\n" NORMAL);
            }
        } else
        {
          fprintf(stderr,RED "approximateBodyFromMotionBufferUsingInverseKinematics could not identify part `%s`\n" NORMAL,bodyPart);
        }

  return finalMAEInPixels;
}


int bvhConverter_smooth(BVHHandle ctxHandle,int frameID,float fSampling,float fCutoff)
{
 struct BVHContext *ctx = CTX(ctxHandle);
 if (ctx->solution==0) { fprintf(stderr,RED "bvhConverter_smooth has no solution to work with..\n" NORMAL);           return 0; }
 if (ctx->filter==0)   { fprintf(stderr,RED "bvhConverter_smooth has no initialized filter to work with..\n" NORMAL); return 0; }

 if ( (fSampling>0.0) && (fCutoff>0.0) )
   {
    //Only perform smoothing if sampling/cutoff is set..
    //fprintf(stderr,GREEN "bvhConverter_smooth going through motions\n" NORMAL); //<- reduce spam
    for (int mID=0; mID<ctx->solution->bufferSize; mID++)
               {
                   ctx->solution->motion[mID] = butterWorth_filterArrayElement(ctx->filter,mID,ctx->solution->motion[mID]);
               }
    //fprintf(stderr,GREEN "copyback..\n" NORMAL); //<- reduce spam


    if(!bvh_copyMotionBufferToMotionFrame(
                                          &ctx->motion,
                                          frameID,
                                          ctx->solution
                                         )
                )
                {
                    fprintf(stderr,RED "Failed bvh_copyMotionBufferToMotionFrame\n" NORMAL);
                }

     //Perform and update projections for new results..!
     bvhConverter_processFrame(ctxHandle,frameID);
     return 1;
   }
  return 0;
}



int bvhConverter(int argc,const char **argv)
{
    fprintf(stderr,RED "BVHConverter.c main is a stub please use the python code\n" NORMAL);
    return 0;
}


BVHHandle bvh_createContext()
{
    return calloc(1, sizeof(struct BVHContext));
}

void bvh_destroyContext(BVHHandle ctxHandle)
{
    if (!ctxHandle) { return; }
    struct BVHContext *ctx = CTX(ctxHandle);
    if (ctx->face)        { free(ctx->face);        ctx->face        = NULL; }
    if (ctx->body)        { free(ctx->body);         ctx->body        = NULL; }
    if (ctx->lhand)       { free(ctx->lhand);        ctx->lhand       = NULL; }
    if (ctx->rhand)       { free(ctx->rhand);        ctx->rhand       = NULL; }
    if (ctx->penultimate) { free(ctx->penultimate);  ctx->penultimate = NULL; }
    if (ctx->previous)    { free(ctx->previous);     ctx->previous    = NULL; }
    if (ctx->solution)    { free(ctx->solution);     ctx->solution    = NULL; }
    if (ctx->filter)      { free(ctx->filter);       ctx->filter      = NULL; }
    free(ctx);
}



int main(int argc,const char **argv)
{
    fprintf(stderr,RED "BVHConverter.c main is a stub please use the python code\n" NORMAL);
    return 0;
}
