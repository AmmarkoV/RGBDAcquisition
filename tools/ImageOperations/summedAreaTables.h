/** @file summedAreaTables.h
 *  @brief A simple implementation of a mean filter that uses summed area tables ( integral images )
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef SUMMEDAREATABLES_H_INCLUDED
#define SUMMEDAREATABLES_H_INCLUDED


/**@brief Allocate and output an integral image of a source rgb image
 * @ingroup summedareatables

 * @param Pointer to source image in floating point encoding
 * @param width of source  image
 * @param height of source image

 * @retval Pointer to memory output,0=error */
unsigned int * generateSummedAreaTableRGB(unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight );


/**
 * @brief Output is the mean filter
 * @ingroup mean

 * @param Pointer to target image in floating point encoding
 * @param width of target image
 * @param height of target image
 * @param channels of target image


 * @param Pointer to source image in floating point encoding
 * @param width of source  image
 * @param height of source image
 * @param channels of source image

 * @param width of median filter kernel
 * @param height of median filter kernel

 * @retval 1=success,0=error
 */

int meanFilterSAT(
                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels ,
                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels ,
                  unsigned int blockWidth , unsigned int blockHeight
                 );

#endif // SUMMEDAREATABLES_H_INCLUDED
