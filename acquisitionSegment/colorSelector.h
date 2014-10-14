/** @file colorSelector.h
 *  @brief The code that segments color and populates a bitmap with selected/deselected pixels
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef COLORSELECTOR_H_INCLUDED
#define COLORSELECTOR_H_INCLUDED


#include "AcquisitionSegment.h"
#include "../tools/Calibration/calibration.h"


/**
 * @brief  Allocates and returns a buffer that is width*height and contains 0 when a pixel is to be segmented and 1 where it should be kept
 * @ingroup acquisitionSegment
 * @param source , A pointer to an RGB frame
 * @param width , Width dimension of RGB frame
 * @param height , Height dimension of RGB frame
 * @param segConf , Criteria to use for segmentation of the RGB frame
 * @param calib , Calibration information for the specific image
 * @param Output Value describing the number of pixels selected from this filter
 * @retval  Pointer to the a bitmap  that describes selected pixels (you shouldnt manually free it) , 0 = Failure
 * @bug   Pixel flood pre needs to be reconnected
 */
unsigned char * selectSegmentationForRGBFrame( unsigned char * source ,
                                               unsigned int width ,
                                               unsigned int height ,
                                               struct SegmentationFeaturesRGB * segConf,
                                               struct calibration * calib,
                                               unsigned int * selectedPixels
                                             );

#endif // COLORSELECTOR_H_INCLUDED
