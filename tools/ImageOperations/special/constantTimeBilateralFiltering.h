/** @file constantTimeBilateralFiltering.h
 *  @brief This is an implementation of the constant time bilateral filtering technique
 *   published as :
     Constant Time Median and Bilateral Filtering
     Qingxiong Yang  Narendra Ahuja Kar-Han Tan..
     The implementation tries can use the deriche gaussian blur to further speed up
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef CONSTANTTIMEBILATERALFILTERING_H_INCLUDED
#define CONSTANTTIMEBILATERALFILTERING_H_INCLUDED



/**
 * @brief An implementation of constant time bilateral filtering
 * @ingroup specialFilters

 * @param Pointer to source  image
 * @param width of source  image
 * @param height of source image
 * @param channels of source image ( 1 = grayscale , 3 = rgb )

 * @param Pointer to target image
 * @param width of target image
 * @param height of target image

 * @param Pointer to sigma to use for the algorithm
 * @param Number of bins to use for the algorithm
 * @param Selector of method to use ( 1=Deriche recursive Gaussian , 0=Gaussian box filter , other value=original image )

 * @bug constantTimeBilateralFilter only works for 1channel images

 * @retval 1=success,0=error
 */
int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float * sigma ,
                                unsigned int bins ,
                                int useDeriche
                               );

#endif // CONSTANTTIMEBILATERALFILTERING_H_INCLUDED
