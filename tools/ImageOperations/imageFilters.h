#ifndef IMAGEFILTERS_H_INCLUDED
#define IMAGEFILTERS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include "../ImagePrimitives/image.h"

float * allocateGaussianKernel(unsigned int dimension);
int monochrome(struct Image * img);
int contrast(struct Image * img,float scaleValue);


#ifdef __cplusplus
}
#endif

#endif // IMAGEFILTERS_H_INCLUDED
