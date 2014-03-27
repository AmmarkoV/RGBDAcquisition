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
   if (RGB==0)
   {
       fprintf(stderr,"segmentRGBAndDepthFrame called with a  null RGB frame \n");
       return 0;
   }
   if (Depth==0)
   {
       fprintf(stderr,"segmentRGBAndDepthFrame called with a  null Depth frame \n");
       return 0;
   }

  if (segConfDepth->autoPlaneSegmentation)
   {
     automaticPlaneSegmentation(Depth,width,height,segConfDepth,calib);
   }


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
  if (combinationMode == COMBINE_SWAP )
  {
     //If we want to swap RGB and Depth  we just swap it
     executeSegmentationRGB(RGB,selectedDepth,width,height,segConfRGB);
     //The same goes for Depth
     executeSegmentationDepth(Depth,selectedRGB,width,height,segConfDepth);
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
      if (combinedSelection!=0) { free(combinedSelection); combinedSelection=0; }
     }
  }

  //Free memory from selections..
  if (selectedRGB!=0)   { free(selectedRGB);   selectedRGB=0;     }
  if (selectedDepth!=0) { free(selectedDepth); selectedDepth=0;   }

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


   segConfRGB->enableRGBMotionDetection=0;
   segConfRGB->firstRGBFrame=0;
   segConfRGB->firstRGBFrameByteSize=0;
   segConfRGB->motionRThreshold=15;
   segConfRGB->motionGThreshold=15;
   segConfRGB->motionBThreshold=15;

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

   segConfDepth->enableDepthMotionDetection=0;
   segConfDepth->firstDepthFrame=0;
   segConfDepth->firstDepthFrameByteSize=0;
   segConfDepth->motionDistanceThreshold=25;


   segConfDepth->enablePlaneSegmentation=0;
   int i=0;
   for (i=0; i<3; i++) { segConfDepth->p1[i]=0.0; segConfDepth->p2[i]=0.0; segConfDepth->p3[i]=0.0; }

  return 1;
}


int copyRGBSegmentation(struct SegmentationFeaturesRGB* target, struct SegmentationFeaturesRGB* source)
{
   memcpy(target,source,sizeof(struct SegmentationFeaturesRGB));
   return 1;
}


int copyDepthSegmentation(struct SegmentationFeaturesDepth* target, struct SegmentationFeaturesDepth* source)
{
   memcpy(target,source,sizeof(struct SegmentationFeaturesDepth));
   return 1;
}

int printDepthSegmentationData(char * label , struct SegmentationFeaturesDepth * dat)
{
  fprintf(stderr,"\n\n\n------------------------------------------------------\n");
  fprintf(stderr,"%s \n",label);
  fprintf(stderr,"------------------------------------------------------\n");
  fprintf(stderr,"Printout of configuration data for depth segmentation\n");

  fprintf(stderr,"Depth min %u max %u\n",dat->minDepth,dat->maxDepth);
  fprintf(stderr,"Depth crop Values %u,%u -> %u,%u \n",dat->minX,dat->minY,dat->maxX,dat->maxY);

  fprintf(stderr,"Total depth flood fill points %u \n",dat->floodErase.totalPoints);

  fprintf(stderr,"Depth Bounding Box State : %u \n",dat->enableBBox);
  fprintf(stderr,"Depth Bounding Box %0.2f,%0.2f,%0.2f -> %0.2f,%0.2f,%0.2f\n",dat->bboxX1,dat->bboxY1,dat->bboxZ1,dat->bboxX2,dat->bboxY2,dat->bboxZ2);

  return 1;
}



int segmentGetDepthBlobAverage(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                               unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                               float * centerX , float * centerY , float * centerZ)
{
  return getDepthBlobAverage(frame,frameWidth,frameHeight,sX,sY,width,height,centerX,centerY,centerZ);
}



int saveSegmentationDataToFile(char* filename , struct SegmentationFeaturesRGB * rgbSeg , struct SegmentationFeaturesDepth * depthSeg , unsigned int combinationMode)
{
    fprintf(stderr,"TODO:  Add all segmentation data here..\n\n");

    int i=0;
    FILE * fp;
    fp  = fopen(filename,"w");
    if(fp!=0)
    {

      if (depthSeg->floodErase.totalPoints>0)
      {
        for (i=0; i<depthSeg->floodErase.totalPoints; i++)
        {
          fprintf(fp,"-floodEraseDepthSource %u %u %u \n",
                    depthSeg->floodErase.pX[i],
                    depthSeg->floodErase.pY[i],
                    depthSeg->floodErase.threshold[i]);
        }
      }



      if ( depthSeg->enableBBox )
      {
          fprintf(fp,"-bbox %f %f %f %f %f %f \n",
                       depthSeg->bboxX1,depthSeg->bboxY1,depthSeg->bboxZ1 ,
                       depthSeg->bboxX2,depthSeg->bboxY2,depthSeg->bboxZ2);
      }

      if ( depthSeg->enablePlaneSegmentation )
      {
          fprintf(fp,"-plane %f %f %f %f %f %f %f %f %f\n",
                       depthSeg->p1[0],depthSeg->p1[1],depthSeg->p1[2] ,
                       depthSeg->p2[0],depthSeg->p2[1],depthSeg->p2[2] ,
                       depthSeg->p3[0],depthSeg->p3[1],depthSeg->p3[2]  );
      }

      if ( depthSeg->enableDepthMotionDetection )
      {
          fprintf(fp,"-depthMotion %u \n",depthSeg->motionDistanceThreshold);
      }


     fprintf(fp,"-cropDepth %u %u %u %u\n",depthSeg->minX,depthSeg->minY,depthSeg->maxX,depthSeg->maxY);


     fprintf(fp,"-minDepth %f\n",depthSeg->minDepth);
     fprintf(fp,"-maxDepth %f\n",depthSeg->maxDepth);


     fprintf(fp,"-cropRGB %u %u %u %u\n",rgbSeg->minX,rgbSeg->minY,rgbSeg->maxX,rgbSeg->maxY);


     fprintf(fp,"-minRGB %u %u %u\n",rgbSeg->minR,rgbSeg->minG,rgbSeg->minB);
     fprintf(fp,"-maxRGB %u %u %u\n",rgbSeg->maxR,rgbSeg->maxG,rgbSeg->maxB);


     fprintf(fp,"-eraseRGB %u %u %u\n",rgbSeg->eraseColorR,rgbSeg->eraseColorG,rgbSeg->eraseColorB);

     fprintf(fp,"-replaceRGB %u %u %u\n",rgbSeg->replaceR,rgbSeg->replaceG,rgbSeg->replaceB);

      fclose(fp);
      return 1;
    }
  return 0;
}


int pickCombinationModeFromString(char * str)
{
  if (strcasecmp(str,"and")==0) { return COMBINE_AND; } else
  if (strcasecmp(str,"or")==0)  { return COMBINE_OR; } else
  if (strcasecmp(str,"xor")==0) { return COMBINE_XOR; } else
  if (strcasecmp(str,"rgb")==0) { return COMBINE_KEEP_ONLY_RGB; } else
  if (strcasecmp(str,"depth")==0) { return COMBINE_KEEP_ONLY_DEPTH; }

  fprintf(stderr,"Could not understand combination method %s\n",str);
  return DONT_COMBINE;
}




int loadSegmentationDataFromArgs(int argc, char *argv[] , struct SegmentationFeaturesRGB * rgbSeg , struct SegmentationFeaturesDepth * depthSeg , unsigned int * combinationMode)
{

  int i=0;
  for (i=0; i<argc; i++)
  {

    if (strcmp(argv[i],"-floodEraseDepthSource")==0)
                                                    {
                                                     depthSeg->floodErase.pX[depthSeg->floodErase.totalPoints] = atoi(argv[i+1]);
                                                     depthSeg->floodErase.pY[depthSeg->floodErase.totalPoints] = atoi(argv[i+2]);
                                                     depthSeg->floodErase.threshold[depthSeg->floodErase.totalPoints] = atoi(argv[i+3]);
                                                     depthSeg->floodErase.source=1;
                                                     ++depthSeg->floodErase.totalPoints;
                                                    } else
    if (strcmp(argv[i],"-floodEraseRGBSource")==0)
                                                {
                                                  rgbSeg->floodErase.pX[rgbSeg->floodErase.totalPoints] = atoi(argv[i+1]);
                                                  rgbSeg->floodErase.pY[rgbSeg->floodErase.totalPoints] = atoi(argv[i+2]);
                                                  rgbSeg->floodErase.threshold[rgbSeg->floodErase.totalPoints] = atoi(argv[i+3]);
                                                  rgbSeg->floodErase.source=1;
                                                  ++rgbSeg->floodErase.totalPoints;
                                                } else
    if (strcmp(argv[i],"-floodEraseRGBTarget")==0) {
                                                  rgbSeg->floodErase.pX[rgbSeg->floodErase.totalPoints] = atoi(argv[i+1]);
                                                  rgbSeg->floodErase.pY[rgbSeg->floodErase.totalPoints] = atoi(argv[i+2]);
                                                  rgbSeg->floodErase.threshold[rgbSeg->floodErase.totalPoints] = atoi(argv[i+3]);
                                                  rgbSeg->floodErase.target=1;
                                                  ++rgbSeg->floodErase.totalPoints;
                                                 } else
    if (strcmp(argv[i],"-cropRGB")==0)    { rgbSeg->minX = atoi(argv[i+1]); rgbSeg->minY = atoi(argv[i+2]);
                                            rgbSeg->maxX = atoi(argv[i+3]); rgbSeg->maxY = atoi(argv[i+4]);   } else
    if (strcmp(argv[i],"-cropDepth")==0)  { depthSeg->minX = atoi(argv[i+1]); depthSeg->minY = atoi(argv[i+2]);
                                            depthSeg->maxX = atoi(argv[i+3]); depthSeg->maxY = atoi(argv[i+4]);   } else
    if (strcmp(argv[i],"-minRGB")==0)     { rgbSeg->minR = atoi(argv[i+1]); rgbSeg->minG = atoi(argv[i+2]); rgbSeg->minB = atoi(argv[i+3]);  } else
    if (strcmp(argv[i],"-maxRGB")==0)     { rgbSeg->maxR = atoi(argv[i+1]); rgbSeg->maxG = atoi(argv[i+2]); rgbSeg->maxB = atoi(argv[i+3]);   } else
    if (strcmp(argv[i],"-eraseRGB")==0) { rgbSeg->eraseColorR = atoi(argv[i+1]); rgbSeg->eraseColorG = atoi(argv[i+2]); rgbSeg->eraseColorB = atoi(argv[i+3]);  } else
    if (strcmp(argv[i],"-replaceRGB")==0) { rgbSeg->replaceR = atoi(argv[i+1]); rgbSeg->replaceG = atoi(argv[i+2]); rgbSeg->replaceB = atoi(argv[i+3]); rgbSeg->enableReplacingColors=1; } else
    if (strcmp(argv[i],"-bbox")==0)       {
                                            depthSeg->enableBBox=1;
                                            depthSeg->bboxX1=(double) internationalAtof(argv[i+1]);
                                            depthSeg->bboxY1=(double) internationalAtof(argv[i+2]);
                                            depthSeg->bboxZ1=(double) internationalAtof(argv[i+3]);
                                            depthSeg->bboxX2=(double) internationalAtof(argv[i+4]);
                                            depthSeg->bboxY2=(double) internationalAtof(argv[i+5]);
                                            depthSeg->bboxZ2=(double) internationalAtof(argv[i+6]);
                                          } else
    if (strcmp(argv[i],"-depthMotion")==0)
                                          {
                                              depthSeg->enableDepthMotionDetection=1;
                                              depthSeg->firstDepthFrame=0;
                                              depthSeg->firstDepthFrameByteSize=0;
                                              depthSeg->motionDistanceThreshold=atoi(argv[i+1]);
                                          } else
    if (strcmp(argv[i],"-plane")==0)      {
                                            depthSeg->enablePlaneSegmentation=1;
                                            depthSeg->p1[0]=(double) internationalAtof(argv[i+1]); depthSeg->p1[1]=(double) internationalAtof(argv[i+2]); depthSeg->p1[2]=(double) internationalAtof(argv[i+3]);
                                            depthSeg->p2[0]=(double) internationalAtof(argv[i+4]); depthSeg->p2[1]=(double) internationalAtof(argv[i+5]); depthSeg->p2[2]=(double) internationalAtof(argv[i+6]);
                                            depthSeg->p3[0]=(double) internationalAtof(argv[i+7]); depthSeg->p3[1]=(double) internationalAtof(argv[i+8]); depthSeg->p3[2]=(double) internationalAtof(argv[i+9]);
                                          } else
    if (strcmp(argv[i],"-minDepth")==0)   { depthSeg->minDepth = atoi(argv[i+1]);  } else
    if (strcmp(argv[i],"-maxDepth")==0)   { depthSeg->maxDepth = atoi(argv[i+1]);   } else
    if (strcmp(argv[i],"-combine")==0)    { *combinationMode=pickCombinationModeFromString(argv[i+1]); }
  }


  return 1;
}
