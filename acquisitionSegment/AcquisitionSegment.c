#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "colorSelector.h"
#include "depthSelector.h"

#include "AcquisitionSegment.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h"


unsigned char * segmentRGBFrame(unsigned char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf, struct calibration * calib)
{
  return selectSegmentationForRGBFrame(source,width,height,segConf,calib);
}

unsigned char * segmentDepthFrame(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf, struct calibration * calib)
{
 return selectSegmentationForDepthFrame(source,width,height,segConf,calib);
}


int   segmentRGBAndDepthFrame (    unsigned char * RGB ,
                                   unsigned short * Depth ,
                                   unsigned int width , unsigned int height ,
                                   struct SegmentationFeaturesRGB * segConfRGB ,
                                   struct SegmentationFeaturesDepth * segConfDepth,
                                   struct calibration * calib ,
                                   int combinationMode
                               )
{
  //We have some criteria for segmentation at segConfRGB , and segConfDepth
  //First of all we want to use them and the original frames (RGB,Depth) , and select ( on an abstract array )
  //the areas we want and the areas we don't want ..

  //We select the area for segmentation from RGB frame
  unsigned char * selectedRGB = selectSegmentationForRGBFrame(RGB , width , height , segConfRGB , calib);

  //We select the area for segmentation from Depth frame
  unsigned char * selectedDepth = selectSegmentationForDepthFrame(Depth , width , height , segConfDepth , calib);

  //We may chose to combine , or make a different selection for the RGB and Depth Frame
  if ( combinationMode == DONT_COMBINE )
  {
     //If we dont want to combine them we just execute the selection and RGB will
     //now have the segmented output frame
     executeSegmentationRGB(RGB,selectedRGB,width,height,segConfRGB);
     //The same goes for Depth
     executeSegmentationDepth(Depth,selectedDepth,width,height,segConfDepth);
  } else
  {
     //If we do want to combine the selections "Together" , and there are many ways to do that using
     //the combinationMode switch ( see enum combinationModesEnumerator ) , we first make a new selection
     //and then execute the segmentation using it
     unsigned char *  combinedSelection = combineRGBAndDepthToOutput(selectedRGB,selectedDepth,combinationMode,width,height);
     if (combinedSelection==0)
     {
       fprintf(stderr,"Failed to combine outputs using method %u\nCannot execute segmentation\n",combinationMode);
     } else
     {
      //We use the combinedSelection for both RGB and Depth
      executeSegmentationRGB(RGB,combinedSelection,width,height,segConfRGB);
      executeSegmentationDepth(Depth,combinedSelection,width,height,segConfDepth);

      //And we dont forget to free our memory
      if (combinedSelection!=0)
        {
          free(combinedSelection);
          combinedSelection=0;
        }
     }
  }


  //Free memory from selections..
  if (selectedRGB!=0) { free(selectedRGB); selectedRGB=0; }
  if (selectedDepth!=0) { free(selectedDepth); selectedDepth=0; }

  return 1;
}



int initializeRGBSegmentationConfiguration(struct SegmentationFeaturesRGB * segConfRGB , unsigned int width , unsigned int height )
{
   segConfRGB->floodErase.totalPoints = 0;

   segConfRGB->minX=0;  segConfRGB->maxX=width;
   segConfRGB->minY=0; segConfRGB->maxY=height;

   segConfRGB->minR=0; segConfRGB->minG=0; segConfRGB->minB=0;
   segConfRGB->maxR=256; segConfRGB->maxG=256; segConfRGB->maxB=256;

   segConfRGB->enableReplacingColors=0;
   segConfRGB->replaceR=92; segConfRGB->replaceG=45; segConfRGB->replaceB=36;

  return 1;
}

int initializeDepthSegmentationConfiguration(struct SegmentationFeaturesDepth* segConfDepth , unsigned int width , unsigned int height )
{

   segConfDepth->minX=0;     segConfDepth->maxX=width;
   segConfDepth->minY=0;     segConfDepth->maxY=height;
   segConfDepth->minDepth=0; segConfDepth->maxDepth=32500;


   segConfDepth->enableBBox=0;
   segConfDepth->bboxX1 = -1000;     segConfDepth->bboxY1 = -1000;    segConfDepth->bboxZ1 = -10000;
   segConfDepth->bboxX2 = 1000;      segConfDepth->bboxY2 = 1000;     segConfDepth->bboxZ2 = 10000;

   segConfDepth->enablePlaneSegmentation=0;
   int i=0;
   for (i=0; i<3; i++) { segConfDepth->p1[i]=0.0; segConfDepth->p2[i]=0.0; segConfDepth->p3[i]=0.0; }

  return 1;
}


int copyRGBSegmentation(struct SegmentationFeaturesRGB* target, struct SegmentationFeaturesRGB* source)
{
   memcpy(target,source,sizeof(struct SegmentationFeaturesRGB));
   /*
   target->minDepth = source->minDepth;
   target->maxDepth = source->maxDepth;

   target->minX = source->minX;  target->maxX = source->maxX;
   target->minY = source->minY;  target->maxY = source->maxY;

   target->floodErase.totalPoints=0;
   source->floodErase.totalPoints=0;

   target->enableBBox  = source->enableBBox;
   target->bboxX1  = source->bboxX1;
   target->bboxY1  = source->bboxY1;
   target->bboxZ1  = source->bboxZ1;
   target->bboxX2  = source->bboxX2;
   target->bboxY2  = source->bboxY2;
   target->bboxZ2  = source->bboxZ2;

   target->enablePlaneSegmentation  = source-> ;
   double p1[3];
   double p2[3];
   double p3[3];*/
   return 1;
}


int copyDepthSegmentation(struct SegmentationFeaturesDepth* target, struct SegmentationFeaturesDepth* source)
{
   memcpy(target,source,sizeof(struct SegmentationFeaturesDepth));
   return 1;
}

int printDepthSegmentationData(char * label , struct SegmentationFeaturesDepth * dat)
{
  fprintf(stderr,"%s \n",label);
  fprintf(stderr,"------------------------------------------------------\n");
  fprintf(stderr,"Printout of configuration data for depth segmentation\n");

  fprintf(stderr,"Depth min %u max %u\n",dat->minDepth,dat->maxDepth);
  fprintf(stderr,"Depth crop Values %u,%u -> %u,%u \n",dat->minX,dat->minY,dat->maxX,dat->maxY);

  fprintf(stderr,"Total depth flood fill points %u \n",dat->floodErase.totalPoints);

  fprintf(stderr,"Depth Bounding Box %0.2f,%0.2f,%0.2f -> %0.2f,%0.2f,%0.2f\n",dat->bboxX1,dat->bboxY1,dat->bboxZ1,dat->bboxX2,dat->bboxY2,dat->bboxZ2);

  return 1;
}
