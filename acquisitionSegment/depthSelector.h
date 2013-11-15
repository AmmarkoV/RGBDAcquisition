#ifndef DEPTHSELECTOR_H_INCLUDED
#define DEPTHSELECTOR_H_INCLUDED

#include "AcquisitionSegment.h"
#include "../tools/Calibration/calibration.h"

int removeDepthFloodFillBeforeProcessing(unsigned short * source , unsigned short * target , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf );

#endif // DEPTHSELECTOR_H_INCLUDED
