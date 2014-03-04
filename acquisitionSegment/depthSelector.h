/** @file depthSelector.h
 *  @brief The code that segments depth and populates a bitmap with selected/deselected pixels
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef DEPTHSELECTOR_H_INCLUDED
#define DEPTHSELECTOR_H_INCLUDED

#include "AcquisitionSegment.h"
#include "../tools/Calibration/calibration.h"

int removeDepthFloodFillBeforeProcessing(unsigned short * source , unsigned short * target , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf );



/**
 * @brief  Allocates and returns a buffer that is width*height and contains 0 when a pixel is to be segmented and 1 where it should be kept
 * @ingroup acquisitionSegment
 * @param source , A pointer to a Depth frame
 * @param width , Width dimension of Depth frame
 * @param height , Height dimension of Depth frame
 * @param segConf , Criteria to use for segmentation of the Depth frame
 * @param calib , Calibration information for the specific image
 * @retval  Pointer to the a bitmap  that describes selected pixels (you shouldnt manually free it) , 0 = Failure
 * @bug   Pixel flood pre needs to be reconnected
 */
unsigned char * selectSegmentationForDepthFrame(
                                                 unsigned short * source ,
                                                 unsigned int width ,
                                                 unsigned int height ,
                                                 struct SegmentationFeaturesDepth * segConf ,
                                                 struct calibration * calib
                                                );

#endif // DEPTHSELECTOR_H_INCLUDED
