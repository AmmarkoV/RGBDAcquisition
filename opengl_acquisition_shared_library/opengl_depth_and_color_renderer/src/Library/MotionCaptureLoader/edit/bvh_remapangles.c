#include "bvh_remapangles.h"
#include "../calculate/bvh_transform.h"
#include "../ik/bvh_inverseKinematics.h"
#include "../export/bvh_to_svg.h"

#include <stdio.h>
#include <math.h>


/*
//THIS IS NOT USED ANYWHERE
float bvh_constrainAngleCentered180(float angle)
{
   angle = fmod(angle,360.0);
   if (angle<0.0)
     { angle+=360.0; }
   return angle;
}
*/


// We have circles A , B and C and we are trying to map circles A and B to circle C
// Because neural networks get confused when coordinates jump from 0 to 360
//
//                                -360 A 0
//
//                                   0 B 360
//
//                                -180 C 180
//
//
//      -270A . 90B . -90C             *                 90C  270B  -90A
//
//
//                                  -1 C 1
//
//                                 179 B 181
//
//                                -181 C -179
//
//
//We want to add 180 degrees to the model so 0 is oriented towards us..!
float bvh_constrainAngleCentered0(float angle,unsigned int flipOrientation)
{
    float angleFrom_minus360_to_plus360;
    float angleRotated = angle+180;

     if (angleRotated<0.0)
     {
       angleFrom_minus360_to_plus360 = (-1*fmod(-1*(angleRotated),360.0))+180;
     } else
     {
       angleFrom_minus360_to_plus360 = (fmod((angleRotated),360.0))-180;
     }

    //If we want to flip orientation we just add or subtract 180 depending on the case
    //To retrieve correct orientatiation we do the opposite
    if (flipOrientation)
    {
      if (angleFrom_minus360_to_plus360<0.0) { angleFrom_minus360_to_plus360+=180.0; } else
      if (angleFrom_minus360_to_plus360>0.0) { angleFrom_minus360_to_plus360-=180.0; } else
                                             { angleFrom_minus360_to_plus360=180.0;  }
    }

   return angleFrom_minus360_to_plus360;
}



float bvh_RemapAngleCentered0(float angle, unsigned int constrainOrientation)
{
    /*
   float angleShifted = angle;
   //We want to add 180 degrees to the model so 0 is oriented towards us..!
   switch (constrainOrientation)
   {
      case BVH_ENFORCE_NO_ORIENTATION :                          return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_FRONT_ORIENTATION :                       return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_BACK_ORIENTATION :                        return bvh_constrainAngleCentered0(angleShifted,1); break;
      case BVH_ENFORCE_LEFT_ORIENTATION :    angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_RIGHT_ORIENTATION :   angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,1); break;
   };
*/
   fprintf(stderr,"bvh_RemapAngleCentered0: Did not change angle using deprecated code..\n");
  return angle;
}



int bvh_swapJointRotationAxis(struct BVH_MotionCapture * bvh,char inputRotationOrder,char swappedRotationOrder)
{
  if ( (bvh!=0) && (bvh->jointHierarchy!=0) )
  {
    for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
    {
        if (bvh->jointHierarchy[jID].channelRotationOrder == inputRotationOrder)
        {
          bvh->jointHierarchy[jID].channelRotationOrder = swappedRotationOrder;
          //fprintf(stderr,"swapping jID %u (%s) from %u to %u\n",jID,bvh->jointHierarchy[jID].jointName,inputRotationOrder,swappedRotationOrder);
        }
    }

    return 1;
  }

 return 0;
}


int bvh_swapJointNameRotationAxis(struct BVH_MotionCapture * bvh,const char * jointName,char inputRotationOrder,char swappedRotationOrder)
{
  if ( (bvh!=0) && (bvh->jointHierarchy!=0) )
  {
    BVHJointID jID=0;
    //----------------
    if ( bvh_getJointIDFromJointName(bvh,jointName,&jID) )
    {
      if (bvh->jointHierarchy[jID].channelRotationOrder == inputRotationOrder)
        {
          bvh->jointHierarchy[jID].channelRotationOrder = swappedRotationOrder;
          //fprintf(stderr,"swapping jID %u (%s) from %u to %u\n",jID,bvh->jointHierarchy[jID].jointName,inputRotationOrder,swappedRotationOrder);
          return 1;
        }
    }
  }

 return 0;
}



float meanBVH2DDistanceStudy(
                             struct BVH_MotionCapture * mc,
                             struct BVH_Transform * bvhSourceTransform,
                             struct BVH_Transform * bvhTargetTransform
                            )
{
 if (mc==0)                 { return 0.0; }
 if (bvhSourceTransform==0) { return 0.0; }
 if (bvhTargetTransform==0) { return 0.0; }
 //-----------------------------------------

 if ( (bvhSourceTransform->joint!=0) && (bvhTargetTransform->joint!=0) )
 {
  float sumOf2DDistances=0.0;
  unsigned int numberOfSamples=0;
  for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
        {
          //fprintf(stderr,"meanBVH2DDistanceStudy(jID -> %u)\n",jID);
          float sX=bvhSourceTransform->joint[jID].pos2D[0];
          float sY=bvhSourceTransform->joint[jID].pos2D[1];
          float tX=bvhTargetTransform->joint[jID].pos2D[0];
          float tY=bvhTargetTransform->joint[jID].pos2D[1];

          if ( ( (sX!=0.0) || (sY!=0.0) ) && ( (tX!=0.0) || (tY!=0.0) ) )
                {
                    float this2DDistance=get2DPointDistance(sX,sY,tX,tY);
                    //fprintf(stderr,"meanBVH2DDistanceStudy(jID -> %u) => %f\n",jID,this2DDistance);
                    numberOfSamples+=1;
                    sumOf2DDistances+=this2DDistance;
                }
        }

        if (numberOfSamples>0)
        {
            return (float)  sumOf2DDistances/numberOfSamples;
        }
 }
 //-----------------------------------------
  return 0.0;
}

int convertHeatmapToProbabilities(
                                  float * output,
                                  unsigned int heatmapResolution
                                 )
{
 float scale = 100.0; // Scale to a 100% to  make CSV file better
 float max = 0.0;
 float sum = 0.0;
 //Gather stats
 for (int h=0; h<heatmapResolution; h++)
 {
   if (output[h]>max) { max=output[h]; }
   sum+=output[h];
 }
 //------------
 if (sum!=0.0)
 {
  //Rescale Output
  for (int h=0; h<heatmapResolution; h++)
  {
    output[h] = (scale * (max-output[h])) / sum;
  }
 return 1;
 }
 //------------
 return 0;
}

int countBodyDoF(struct BVH_MotionCapture * mc)
{
  int isJointSelected=1;
  int isJointEndSiteSelected=1;
  int count=0;

  for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
         {
            for (unsigned int channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
                 {
                     count+=1;
                 }
         }
       }
   return count;
}

int countNumberOfHeatmapResolutions(
                                    struct BVH_MotionCapture * mc,
                                    float *rangeMinimum,
                                    float *rangeMaximum,
                                    float *resolution
                                   )
{
  int count=0;
  float increment = *resolution;
  float v = *rangeMinimum;
  while (v<*rangeMaximum)
        {
          count+=1;
          v+=increment;
        }
  return count;
}

int initializeStandaloneHeatmapFile(
                                    const char * filename,
                                    float *rangeMinimum,
                                    float *rangeMaximum,
                                    float *resolution
                                   )
{
 FILE * fp = fopen(filename,"w");
 if (fp!=0)
         {
          char comma=' ';
          float increment = *resolution;
          float v = *rangeMinimum;
          while (v<*rangeMaximum)
          {
            if (comma==',') { fprintf(fp,",");  } else { comma=','; }
            fprintf(fp,"%0.2f",v);
            v+=increment;
          }

          fprintf(fp,"\n");
          fclose(fp);
          return 1;
         }
 return 0;
}





//  ./BVHTester --printparams --haltonerror --from Motions/05_01.bvh --angleheatmap --filtergimballocks 4 --selectJoints 1 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --hide2DLocationOfJoints 0 8 abdomen chest eye.r eye.l toe1-2.r toe5-3.r toe1-2.l toe5-3.l --perturbJointAngles 2 30.0 rshoulder lshoulder --perturbJointAngles 2 16.0 relbow lelbow --perturbJointAngles 2 10.0 abdomen chest --perturbJointAngles 2 30.0 rhip lhip --perturbJointAngles 4 10.0 lknee rknee lfoot rfoot --perturbJointAngles 2 10.0 abdomen chest --repeat 0 --sampleskip 2 --filterout 0 0 -130.0 0 90 0 1920 1080 570.7 570.3 6 rhand lhip 0 120 rhand rhip 0 120 rhand lhand 0 150 lhand rhip 0 120 lhand lhip 0 120 lhand rhand 0 150 --randomize2D 900 4500 -45 -179.999999 -45 45 180 45 --occlusions --csv debug/ body_all.csv 2d+3d+bvh
int dumpBVHAsProbabilitiesHeader(
                                 struct BVH_MotionCapture * mc,
                                 const char * filename,
                                 float *rangeMinimum,
                                 float *rangeMaximum,
                                 float *resolution
                                )
{
     char initialFilenameWithoutExtension[512]={0};
     snprintf(initialFilenameWithoutExtension,512,"%s",filename);
     char * dot = strchr(initialFilenameWithoutExtension,'.');
     if (dot!=0) { *dot=0; }
     //----------------------------------------------------------
     char specificJointFilename[1024]={0};
     //----------------------------------------------------------

     int isJointSelected=1;
     int isJointEndSiteSelected=1;

     for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
       {
          bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
         {
            unsigned int channelID=0;
            for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
                 {
                    snprintf(
                             specificJointFilename,1024,"%s_%s_%s.csv",
                             initialFilenameWithoutExtension,
                             mc->jointHierarchy[jID].jointNameLowercase,
                             channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]
                            );
                     //----------------------------------------------------------
                     initializeStandaloneHeatmapFile(specificJointFilename,rangeMinimum,rangeMaximum,resolution);
                     //----------------------------------------------------------
                 }
         }
       }

   return 1;
}



int generateHeatmap(
                     float * output,
                     unsigned int heatmapResolution,
                     struct BVH_MotionCapture * bvh,
                     struct simpleRenderer * renderer,
                     BVHFrameID fID,
                     BVHMotionChannelID mIDRelativeToOneFrame,
                     float *rangeMinimum,
                     float *rangeMaximum,
                     float *resolution,
                     int doSVG
                   )
{
 if (output==0) { return 0; }
 //-----------------------------------------------------------------------------------
 struct BVH_Transform bvhTransformOriginal = {0};
 struct BVH_Transform bvhTransformChanged  = {0};
 //-----------------------------------------------------------------------------------
 BVHMotionChannelID mID = (fID * bvh->numberOfValuesPerFrame) + mIDRelativeToOneFrame;
 float originalValue = bvh_getMotionValue(bvh,mID);
 //-----------------------------------------------------------------------------------
 if (doSVG)
 {
 fprintf(stderr,"generateHeatmap(%u,%u,%0.2f,%0.2f)\n",fID,mIDRelativeToOneFrame,*rangeMinimum,*rangeMaximum);
 fprintf(stderr,"MID %u / Joint %u/ Channel %u / %s %s / Original Value %0.2f / Min %0.2f / Max %0.2f / Increment %0.2f\n",
           mID,
           bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID,
           bvh->motionToJointLookup[mIDRelativeToOneFrame].channelID,
           bvh->jointHierarchy[bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID].jointName,
           channelNames[bvh->motionToJointLookup[mIDRelativeToOneFrame].channelID],
           originalValue,
           *rangeMinimum,
           *rangeMaximum,
           *resolution
          );
 }
 //-----------------------------------------------------------------------------------
 if (
       (bvh_loadTransformForFrame(bvh,fID,&bvhTransformOriginal,0)) &&
       (bvh_projectTo2D(bvh,&bvhTransformOriginal,renderer,0,0))
    )
    {
       if (doSVG)
       {
        dumpBVHToSVGFrame(
                          "study.svg",
                          bvh,
                          &bvhTransformOriginal,
                          fID,
                          renderer
                         );
       }

       unsigned int value = 0;
       float increment = *resolution;
       float v         = *rangeMinimum;
       while (v<*rangeMaximum)
        {
          bvh_setMotionValue(bvh,mID,&v);
          //-----------------------------
          if (
              (bvh_loadTransformForFrame(bvh,fID,&bvhTransformChanged,0)) &&
              (bvh_projectTo2D(bvh,&bvhTransformChanged,renderer,0,0))
             )
             {
                 output[value] = meanBVH2DDistanceStudy(
                                                        bvh,
                                                        &bvhTransformChanged,
                                                        &bvhTransformOriginal
                                                       );
             }
          v+=increment;
          value+=1;
        }
        //-----------------------------------------
        convertHeatmapToProbabilities(
                                       output,
                                       heatmapResolution
                                     );
        //-----------------------------------------
        bvh_setMotionValue(bvh,mID,&originalValue);
        return 1;
      } else
      {
          fprintf(stderr,"Failed projecting original..\n");
      }
  return 0;
}



int bvh_plotJointChannelHeatmap(
                                 const char * filename,
                                 struct BVH_MotionCapture * bvh,
                                 struct simpleRenderer * renderer,
                                 BVHFrameID fID,
                                 BVHJointID jID,
                                 BVHMotionChannelID channelID,
                                 float *rangeMinimum,
                                 float *rangeMaximum,
                                 float *resolution,
                                 unsigned int heatmapResolution
                               )
{
  FILE * fp = fopen(filename,"a");
  if (fp!=0)
  {
   float * output = (float*) malloc(sizeof(float) * heatmapResolution);
   if (output!=0)
   {
    struct BVH_Transform bvhTransformOriginal = {0};
    struct BVH_Transform bvhTransformChanged  = {0};
    //----------------------------------------------------------
    BVHMotionChannelID mIDRelativeToOneFrame = bvh->jointToMotionLookup[jID].channelIDMotionOffset[channelID];
    //----------------------------------------------------------
    BVHMotionChannelID mID = (fID * bvh->numberOfValuesPerFrame) + mIDRelativeToOneFrame;
    //----------------------------------------------------------


    if (
         generateHeatmap(
                         output,
                         heatmapResolution,
                         bvh,
                         renderer,
                         fID,
                         mIDRelativeToOneFrame,
                         rangeMinimum,
                         rangeMaximum,
                         resolution,
                         0 // Dont dump SVG
                        )
       )
       {
           char comma=' ';
           for (int h=0; h<heatmapResolution; h++)
           {
             //-----------------------------------------------------------------------------
             if (comma==',') { fprintf(fp,","); } else { comma=','; }
             //-----------------------------------------------------------------------------
             fprintf(fp,"%0.2f",output[h]);
             //-----------------------------------------------------------------------------
           }
       }

      if (output!=0) { free(output); }
      }
      fprintf(fp,"\n");
      fclose(fp);
      return 1;
  }
  return 0;
}


int dumpBVHAsProbabilitiesBody(
                                 struct BVH_MotionCapture * mc,
                                 const char * filename,
                                 struct simpleRenderer * renderer,
                                 BVHFrameID fID,
                                 int numberOfHeatmapTasks,
                                 int heatmapResolution,
                                 float *rangeMinimum,
                                 float *rangeMaximum,
                                 float *resolution
                             )
{
  if ( (rangeMinimum==0) || (rangeMaximum==0) || (resolution==0) ) { fprintf(stderr,"dumpBVHAsProbabilitiesBody no ranges\n"); return 0; }

  char initialFilenameWithoutExtension[512]={0};
  snprintf(initialFilenameWithoutExtension,512,"%s",filename);
  char * dot = strchr(initialFilenameWithoutExtension,'.');
  if (dot!=0) { *dot=0; }
  //----------------------------------------------------------
  char specificJointFilename[1024]={0};
  //----------------------------------------------------------

  int isJointSelected=1;
  int isJointEndSiteSelected=1;
  int executedTasks=0;

  for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
         {
            for (unsigned int channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
                 {
                    //fprintf(stderr,"\r Generation %u/%u %0.2f%%\r",executedTasks,numberOfHeatmapTasks,(float) (100*executedTasks)/numberOfHeatmapTasks);
                    executedTasks+=1;
                    //----------------------------------------------------------
                    snprintf(
                             specificJointFilename,1024,"%s_%s_%s.csv",
                             initialFilenameWithoutExtension,
                             mc->jointHierarchy[jID].jointNameLowercase,
                             channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]
                            );
                     //-----------------------------------------------------------------------------
                     bvh_plotJointChannelHeatmap(
                                                 specificJointFilename,
                                                 mc,
                                                 renderer,
                                                 fID,
                                                 jID,
                                                 channelID,
                                                 rangeMinimum,
                                                 rangeMaximum,
                                                 resolution,
                                                 heatmapResolution
                                                );
                     //----------------------------------------------------------
                 }
         }
         //-----------------------------------------------------------------------------
       }

   return 1;
}






int bvh_studyMID2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHMotionChannelID mIDRelativeToOneFrame,
                           float *rangeMinimum,
                           float *rangeMaximum,
                           float *resolution
                          )
{
  //Declare and populate the simpleRenderer that will project our 3D points
  struct simpleRenderer renderer={0};

  if (renderingConfiguration->isDefined)
  {
    renderer.fx     = renderingConfiguration->fX;
    renderer.fy     = renderingConfiguration->fY;
    renderer.skew   = 1.0;
    renderer.cx     = renderingConfiguration->cX;
    renderer.cy     = renderingConfiguration->cY;
    renderer.near   = renderingConfiguration->near;
    renderer.far    = renderingConfiguration->far;
    renderer.width  = renderingConfiguration->width;
    renderer.height = renderingConfiguration->height;

    simpleRendererInitializeFromExplicitConfiguration(&renderer);
    //bvh_freeTransform(&bvhTransform);

    fprintf(stderr,"Direct Rendering is not implemented yet, please don't use it..\n");
    return 0;
  } else
  {
   //This is the normal rendering where we just simulate our camera center
   simpleRendererDefaults(
                          &renderer,
                          renderingConfiguration->width,
                          renderingConfiguration->height,
                          renderingConfiguration->fX,
                          renderingConfiguration->fY
                         );
    simpleRendererInitialize(&renderer);
  }

  FILE * fp = fopen("study.dat","w");

  if (fp!=0)
  {
   unsigned int numberOfHeatmapTasks = countNumberOfHeatmapResolutions(bvh,rangeMinimum,rangeMaximum,resolution);
   float * output = (float*) malloc(sizeof(float) * numberOfHeatmapTasks);
   if (output!=0)
   {
    if (
         generateHeatmap(
                         output,
                         numberOfHeatmapTasks,
                         bvh,
                         &renderer,
                         fID,
                         mIDRelativeToOneFrame,
                         rangeMinimum,
                         rangeMaximum,
                         resolution,
                         1 // Dump SVG
                        )
       )
       {
        float increment = *resolution;
        float v = *rangeMinimum;
        for (int h=0; h<numberOfHeatmapTasks; h++)
           {
                 fprintf(fp,"%f %f\n",v,output[h]);
                 v+=increment;
           }
       }
   }
   //File needs to be dumped before gnuplot
   //---------
   fclose(fp);
   //---------

   //using ls 1 t 'TTT'
   char command[2048]={0};
   snprintf(
               command,2048,"gnuplot -e \"set terminal png size 800,512 font 'Helvetica,14'; set output 'out.png'; set xrange[%0.2f:%0.2f]; set style line 1 lt 1 lc rgb 'blue' lw 3; plot 'study.dat' with lines ls 1 title '%s %s Error'\"",
               *rangeMinimum,*rangeMaximum,
               bvh->jointHierarchy[bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID].jointName,
               channelNames[bvh->motionToJointLookup[mIDRelativeToOneFrame].channelID]
           );

   //fprintf(stderr,"%s\n",command);
   int i = system(command);
   return (i==0);
  }

  return 0;
}











int bvh_study3DJoint2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHJointID jID,
                           float *rangeMinimum,
                           float *rangeMaximum,
                           float *resolution
                          )
{
  //Declare and populate the simpleRenderer that will project our 3D points
  struct simpleRenderer renderer={0};

  if (renderingConfiguration->isDefined)
  {
    renderer.fx     = renderingConfiguration->fX;
    renderer.fy     = renderingConfiguration->fY;
    renderer.skew   = 1.0;
    renderer.cx     = renderingConfiguration->cX;
    renderer.cy     = renderingConfiguration->cY;
    renderer.near   = renderingConfiguration->near;
    renderer.far    = renderingConfiguration->far;
    renderer.width  = renderingConfiguration->width;
    renderer.height = renderingConfiguration->height;

    simpleRendererInitializeFromExplicitConfiguration(&renderer);
    //bvh_freeTransform(&bvhTransform);

    fprintf(stderr,"Direct Rendering is not implemented yet, please don't use it..\n");
    return 0;
  } else
  {
   //This is the normal rendering where we just simulate our camera center
   simpleRendererDefaults(
                          &renderer,
                          renderingConfiguration->width,
                          renderingConfiguration->height,
                          renderingConfiguration->fX,
                          renderingConfiguration->fY
                         );
    simpleRendererInitialize(&renderer);
  }

  FILE * fp = fopen("study.dat","w");

  if (fp!=0)
  {
   struct BVH_Transform bvhTransformOriginal = {0};
   struct BVH_Transform bvhTransformChanged  = {0};

   BVHMotionChannelID mIDRelativeToOneFrame = 0;
   //----------------------------------------------------------
   BVHMotionChannelID mIDA = (fID * bvh->numberOfValuesPerFrame) + mIDRelativeToOneFrame;
   BVHMotionChannelID mIDB = (fID * bvh->numberOfValuesPerFrame) + mIDRelativeToOneFrame;
   BVHMotionChannelID mIDC = (fID * bvh->numberOfValuesPerFrame) + mIDRelativeToOneFrame;
   //----------------------------------------------------------
   int c=0;
   for (int channelNumber=0; channelNumber<bvh->jointHierarchy[jID].loadedChannels; channelNumber++)
                 {
                     fprintf(stderr,"Joint %u / Channel Number %u \n",jID,channelNumber);
                     unsigned int channelTypeID = bvh->jointHierarchy[jID].channelType[channelNumber];
                     //-------------------------------------------------------------------------------------
                     if(c==0) {
                                mIDA = bvh_resolveFrameAndJointAndChannelToMotionID(bvh,jID,fID,channelTypeID);
                                fprintf(stderr,"Channel A %u \n",mIDRelativeToOneFrame,mIDA);
                              } else
                     if(c==1) {
                                mIDB = bvh_resolveFrameAndJointAndChannelToMotionID(bvh,jID,fID,channelTypeID);
                                fprintf(stderr,"Channel B %u \n",mIDRelativeToOneFrame,mIDB);
                              } else
                     if(c==2) {
                                mIDC = bvh_resolveFrameAndJointAndChannelToMotionID(bvh,jID,fID,channelTypeID);
                                fprintf(stderr,"Channel C %u \n",mIDRelativeToOneFrame,mIDC);
                              } else
                              {
                                fprintf(stderr,"Too Many Channels (%u)!\n",bvh->jointHierarchy[jID].loadedChannels);
                              }
                     //----------------------------------------------------------
                     c=c+1;
                 }

   fprintf(stderr,"bvh_studyMID2DImpact(fID %u,jID %u (%s),%u channels,mIDA %u,mIDB %u,mIDC %u,%0.2f,%0.2f)\n",
           fID,
           jID,
           bvh->jointHierarchy[jID].jointName,
           bvh->jointHierarchy[jID].loadedChannels,
           mIDA,mIDB,mIDC,*rangeMinimum,*rangeMaximum);

   float originalValueA = bvh_getMotionValue(bvh,mIDA);
   float originalValueB = bvh_getMotionValue(bvh,mIDB);
   float originalValueC = bvh_getMotionValue(bvh,mIDC);


   if (
       (bvh_loadTransformForFrame(bvh,fID,&bvhTransformOriginal,0)) &&
       (bvh_projectTo2D(bvh,&bvhTransformOriginal,&renderer,0,0))
      )
      {
       dumpBVHToSVGFrame(
                         "study.svg",
                         bvh,
                         &bvhTransformOriginal,
                         fID,
                         &renderer
                        );
        float increment = *resolution;
        float vA = *rangeMinimum;
        float vB = *rangeMinimum;
        float vC = *rangeMinimum;
        while (vC<*rangeMaximum)
        {
         bvh_setMotionValue(bvh,mIDC,&vC);
         vB = *rangeMinimum;
         while (vB<*rangeMaximum)
         {
          bvh_setMotionValue(bvh,mIDB,&vB);
          vA = *rangeMinimum;
          while (vA<*rangeMaximum)
          {
           bvh_setMotionValue(bvh,mIDA,&vA);

           if (
              (bvh_loadTransformForFrame(bvh,fID,&bvhTransformChanged,0)) &&
              (bvh_projectTo2D(bvh,&bvhTransformChanged,&renderer,0,0))
              )
             {
                 float mae = meanBVH2DDistanceStudy(
                                                   bvh,
                                                   &bvhTransformChanged,
                                                   &bvhTransformOriginal
                                                  );
                 fprintf(fp,"%0.2f %0.2f %0.2f %0.2f\n",vA,vB,vC,mae);
             }
           vA+=increment*2;
          }
          vB+=increment*5;
         }
         vC+=increment*4;
        }

        bvh_setMotionValue(bvh,mIDA,&originalValueA);
        bvh_setMotionValue(bvh,mIDB,&originalValueB);
        bvh_setMotionValue(bvh,mIDC,&originalValueC);
      } else
      {
          fprintf(stderr,"Failed projecting original..\n");
      }
      fclose(fp);

      //using ls 1 t 'TTT'
      char command[2048]={0};
      snprintf(
               command,2048,"gnuplot -e \"set terminal png size 800,512 font 'Helvetica,14'; set output 'out.png'; set view 45, 45, 1, 1; splot 'study.dat' using 1:2:3:4 w points pointsize 1 palette pointtype 7 title '%s 3D Error'\"",
               *rangeMinimum,*rangeMaximum,
               bvh->jointHierarchy[bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID].jointName
              );

      //fprintf(stderr,"%s\n",command);
      int i = system(command);
      return (i==0);
  }

  return 0;
}

