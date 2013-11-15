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

      fprintf(stderr,"Segmentation executed!\n");
      //And we dont forget to free our memory
      if (combinedSelection!=0) { free(combinedSelection); combinedSelection=0; }
     }
  }


  //Free memory from selections..
  if (selectedRGB!=0) { free(selectedRGB); selectedRGB=0; }
  if (selectedDepth!=0) { free(selectedDepth); selectedDepth=0; }

  return 1;
}
