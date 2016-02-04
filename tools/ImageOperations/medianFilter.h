/** @file medianFilter.h
 *  @brief A simple ( not very efficient ) implementation of a median filter that uses insertion sort
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MEDIANFILTER_H_INCLUDED
#define MEDIANFILTER_H_INCLUDED


/**
 * @brief Apply a median filter over an image
 * @ingroup median

 * @param Pointer to target image in floating point encoding
 * @param width of target image
 * @param height of target image


 * @param Pointer to source image in floating point encoding
 * @param width of source  image
 * @param height of source image

 * @param width of median filter kernel
 * @param height of median filter kernel

 * @retval 1=success,0=error
 */
int medianFilter3ch(
                 unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                 unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                 unsigned int blockWidth , unsigned int blockHeight
                );

#endif // MEDIANFILTER_H_INCLUDED
