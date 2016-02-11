/** @file imageFilters.h
 *  @brief This is a collection of routines
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef IMAGEFILTERS_H_INCLUDED
#define IMAGEFILTERS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include "../Primitives/image.h"


/**
 * @brief Allocate and output a gaussian matrix
 * @ingroup imageMatrix

 * @param Dimension of output gaussian kernel
 * @param Sigma of gaussian matrix
 * @param Do normalization on output kernel

 * @retval Pointer to gaussian kernel,0=error
 */
float * allocateGaussianKernel(unsigned int dimension,float sigma , int normalize);


/**
 * @brief Convert input Image to monochrome
 * @ingroup imageMatrix

 * @param An input image

 * @retval 1=success,0=error
 */
int monochrome(struct Image * img);


/**
 * @brief Change contrast of input Image
 * @ingroup imageMatrix

 * @param An input image
 * @param Scaling value ( 1.0 = no change )

 * @retval 1=success,0=error
 */
int contrast(struct Image * img,float scaleValue);


#ifdef __cplusplus
}
#endif

#endif // IMAGEFILTERS_H_INCLUDED
