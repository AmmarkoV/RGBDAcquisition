#ifndef COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
#define COMBINERGBANDDEPTHOUTPUT_H_INCLUDED

#include "AcquisitionSegment.h"

int executeSegmentationRGB(unsigned char * RGB , unsigned char * selectedRGB , unsigned int width , unsigned int height ,  struct SegmentationFeaturesRGB * segConf );
int executeSegmentationDepth(unsigned short * Depth , unsigned char * selectedDepth , unsigned int width , unsigned int height  );

unsigned char * combineRGBAndDepthToOutput( unsigned char * selectedRGB , unsigned char * selectedDepth , int combinationMode, unsigned int width , unsigned int height );


#endif // COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
