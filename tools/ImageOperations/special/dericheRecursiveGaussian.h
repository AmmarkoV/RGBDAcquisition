/** @file dericheRecursiveGaussian.h
 *  @brief This is an implementation of the deriche recursive gaussian filtering technique
 *   published as :
       Rachid Deriche. Recursively implementating the Gaussian and its derivatives. [Research Report] RR-1893, 1993, pp.24. < inria-00074778 >
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef DERICHERECURSIVEGAUSSIAN_H_INCLUDED
#define DERICHERECURSIVEGAUSSIAN_H_INCLUDED


/**
 * @brief An implementation of the deriche recursive gaussian filtering technique for floating point input/output images
 * @ingroup specialFilters
 * @param timerNumber , the number that specifies what timer we want to start

 * @param Pointer to source image where source image is in floating point format
 * @param width of source  image
 * @param height of source image
 * @param channels of source image ( 1 = grayscale , 3 = rgb )

 * @param Pointer to target image  where target image is in floating point format
 * @param width of target image
 * @param height of target image

 * @param Pointer to sigma to use for the algorithm
 * @param Order of the algorithm ( 0 = normal , 1 = first order derivative , 2 = second order derivative )

 * @bug constantTimeBilateralFilter only works for 1channel images

 * @retval 1=success,0=error
 */
int dericheRecursiveGaussianGrayF(
                                     float * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                     float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                     float *sigma , unsigned int order
                                   );


/**
 * @brief An implementation of the deriche recursive gaussian filtering technique for regular unsigned char input/output images
 * @ingroup specialFilters
 * @param timerNumber , the number that specifies what timer we want to start

 * @param Pointer to source image where source image is in unsigned char format
 * @param width of source  image
 * @param height of source image
 * @param channels of source image ( 1 = grayscale , 3 = rgb )

 * @param Pointer to target image  where target image is in unsigned charformat
 * @param width of target image
 * @param height of target image

 * @param Pointer to sigma to use for the algorithm
 * @param Order of the algorithm ( 0 = normal , 1 = first order derivative , 2 = second order derivative )

 * @bug constantTimeBilateralFilter only works for 1channel images

 * @retval 1=success,0=error
 */
int dericheRecursiveGaussianGray(
                                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                  float *sigma , unsigned int order
                                );


#endif // DERICHERECURSIVEGAUSSIAN_H_INCLUDED
