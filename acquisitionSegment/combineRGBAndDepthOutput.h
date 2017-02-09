#ifndef COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
#define COMBINERGBANDDEPTHOUTPUT_H_INCLUDED

#include "AcquisitionSegment.h"


#ifdef __cplusplus
extern "C"
{
#endif


int dilateSelection(unsigned char * selected , unsigned int width , unsigned int height , unsigned int kernWidth , unsigned int kernHeight , unsigned int kernThreshold);
int erodeSelection(unsigned char * selected , unsigned int width , unsigned int height , unsigned int kernWidth , unsigned int kernHeight , unsigned int kernThreshold);

int invertSelection(unsigned char * selected , unsigned int width , unsigned int height , unsigned int * selectedCount);

int executeSegmentationRGB(unsigned char * RGB , unsigned char * selectedRGB , unsigned int width , unsigned int height ,  struct SegmentationFeaturesRGB * segConf ,unsigned int selectedRGBCount , unsigned int combinationMode);
int executeSegmentationDepth(unsigned short * Depth , unsigned char * selectedDepth , unsigned int width , unsigned int height  ,unsigned int selectedDepthCount , unsigned int combinationMode);

unsigned char * combineRGBAndDepthToOutput( unsigned char * selectedRGB , unsigned char * selectedDepth , int combinationMode, unsigned int width , unsigned int height );

#ifdef __cplusplus
}
#endif

#endif // COMBINERGBANDDEPTHOUTPUT_H_INCLUDED
