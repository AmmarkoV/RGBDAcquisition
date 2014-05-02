/** @file automaticPlaneSegmentation.h
 *  @brief This is a small optimization routine that uses the plane segmentation capabilities to automatically sense a plane using a reference depth image
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug Automatic Plane Segmentation does not work yet , it is under construction
 */


#ifndef AUTOMATICPLANESEGMENTATION_H_INCLUDED
#define AUTOMATICPLANESEGMENTATION_H_INCLUDED

#include "AcquisitionSegment.h"

/**
 * @brief  Automatically detect a plane from an input depth frame
 * @ingroup acquisitionSegment
 * @param Pointer to Depth Input
 * @param Width of RGB Input
 * @param Height of RGB Input
 * @param Structure holding the segmentation results for Depth Frame ( will just update the plane variables )
 * @retval 1=Success,0=Failure
 */
int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , float offset, struct SegmentationFeaturesDepth * segConf , struct calibration * calib );

#endif // AUTOMATICPLANESEGMENTATION_H_INCLUDED
