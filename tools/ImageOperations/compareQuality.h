/** @file compareQuality.h
 *  @brief This is a small routine that outputs the difference in two signals in dB
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef COMPAREQUALITY_H_INCLUDED
#define COMPAREQUALITY_H_INCLUDED


/**
 * @brief Calculate signal difference in dB
 * @ingroup imageMatrix

 * @param Pointer to target image
 * @param width of target image
 * @param height of target image
 * @param channels of target image ( 1 = grayscale , 3 = rgb )


 * @param Pointer to source image
 * @param width of source  image
 * @param height of source image
 * @param channels of source image ( 1 = grayscale , 3 = rgb )



 * @retval difference of two images ,0=no difference on two images or error
 */
float  calculatePSNR(
                         unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels,
                         unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels
                    );

#endif // COMPAREQUALITY_H_INCLUDED
