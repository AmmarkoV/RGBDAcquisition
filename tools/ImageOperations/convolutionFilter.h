/** @file convolutionFilter.h
 *  @brief A simple  convolution algorithm
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef CONVOLUTIONFILTER_H_INCLUDED
#define CONVOLUTIONFILTER_H_INCLUDED


/**
 * @brief Do an image convolution with a floating point input on a 3 channel (RGB) image
 * @ingroup imageMatrix

 * @param Pointer to target image in floating point encoding
 * @param width of target image
 * @param height of target image


 * @param Pointer to source image in floating point encoding
 * @param width of source  image
 * @param height of source image

 * @param Pointer to convolution kernel
 * @param width of convolution kernel
 * @param height of convolution kernel
 * @param divisor to be used with convolution matrix , 1.0 = no divisor

 * @retval difference of two images ,0=no difference on two images or error
 */
int convolutionFilter3ChF(
                       float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                       float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                       float * convolutionMatrix , unsigned int kernelWidth , unsigned int kernelHeight , float divisor
                      );



/**
 * @brief Do an image convolution with a floating point input on a 1 channel (grayscale) image
 * @ingroup imageMatrix

 * @param Pointer to target image in floating point encoding
 * @param width of target image
 * @param height of target image


 * @param Pointer to source image in floating point encoding
 * @param width of source  image
 * @param height of source image

 * @param Pointer to convolution kernel
 * @param width of convolution kernel
 * @param height of convolution kernel
 * @param divisor to be used with convolution matrix , 1.0 = no divisor

 * @retval difference of two images ,0=no difference on two images or error
 */
int convolutionFilter1ChF(
                          float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                          float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          float * convolutionMatrix , unsigned int  kernelWidth , unsigned int kernelHeight , float * divisor
                         );

#endif // CONVOLUTIONFILTER_H_INCLUDED
