#ifndef COLORSELECTOR_H_INCLUDED
#define COLORSELECTOR_H_INCLUDED


#include "AcquisitionSegment.h"
#include "../tools/Calibration/calibration.h"

unsigned char * selectSegmentationForRGBFrame(unsigned char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf, struct calibration * calib);

#endif // COLORSELECTOR_H_INCLUDED
