#ifndef IMAGECONVERSIONS_H_INCLUDED
#define IMAGECONVERSIONS_H_INCLUDED

#include "imageConversions.h"

#include "MovidiusTypes.h"


// 16 bits.  will use this to store half precision floats since C has no
// built in support for it.
typedef unsigned short half;

int loadLabels(const char * filename , struct labelContents * labels );
void *LoadFile(const char *path, unsigned int *length);
half *LoadImage(const char *path, int reqsize, float *mean);
half *LoadImageFromMemory16(const char *buf , unsigned int bufW, unsigned int bufH  , unsigned int bufChans, int reqsize, float *mean);
float *LoadImageFromMemory32(const char *buf , unsigned int bufW, unsigned int bufH  , unsigned int bufChans, int reqsizeX, int reqsizeY, float *mean);


#endif // IMAGECONVERSIONS_H_INCLUDED
