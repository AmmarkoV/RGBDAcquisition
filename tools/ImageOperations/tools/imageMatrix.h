/** @file imageMatrix.h
 *  @brief This is a collection of small routines that do trivial conversions of images and some very easy
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef IMAGEMATRIX_H_INCLUDED
#define IMAGEMATRIX_H_INCLUDED

/**
 * @brief Cast an input image that is encoded in unsigned chars to floating point and normalize values to go from 0-255 to 0-1
 * @ingroup imageMatrix

 * @param Pointer to target floating point encoded image
 * @param Pointer to source unsigned character encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images
 * @retval 1=success,0=error
 */
int castUCharImage2FloatAndNormalize(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);


/**
 * @brief Cast an input image that is encoded in unsigned chars to floating point
 * @ingroup imageMatrix

 * @param Pointer to target floating point encoded image
 * @param Pointer to source unsigned character encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images
 * @retval 1=success,0=error
 */
int castUCharImage2Float(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);


/**
 * @brief Cast an input image that is encoded in floats to  unsigned chars
 * @ingroup imageMatrix

 * @param Pointer to target unsigned character encoded image
 * @param Pointer to source floating point encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images
 * @retval 1=success,0=error
 */
int castFloatImage2UChar(unsigned char * out , float * in, unsigned int width,unsigned int height , unsigned int channels);



/**
 * @brief Allocate a new memory block and output an input image that is encoded in unsigned chars to floating point
 * @ingroup imageMatrix

 * @param Pointer to source unsigned character encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images
 * @retval Pointer to newly allocated floating point image ( needs to be freed manually ),0=error
 */
float * copyUCharImage2Float(unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);


/**
 * @brief Allocate a new memory block and output the product of the division of two unsigned char input images
 * @ingroup imageMatrix

 * @param Pointer to dividend source unsigned character encoded image ( dividend / divisor = quotient  )
 * @param Pointer to divisor source unsigned character encoded image  ( dividend / divisor = quotient  )
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images

 * @retval Pointer to newly allocated unsigned char point image that holds the product of the division,0=error
 */
unsigned char* divideTwoImages(unsigned char *  dividend , unsigned char * divisor , unsigned int width,unsigned int height , unsigned int channels);


/**
 * @brief Output the product of the division of two floating point input images
 * @ingroup imageMatrix

 * @param Pointer to quotient output floating point encoded image ( dividend / divisor = quotient  )
 * @param Pointer to dividend source floating point encoded image ( dividend / divisor = quotient  )
 * @param Pointer to divisor source floating point encoded image  ( dividend / divisor = quotient  )
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images

 * @retval 1=success,0=error
 */
int divide2DMatricesF(float * out , float * dividend , float * divisor , unsigned int width , unsigned int height , unsigned int channels);


/**
 * @brief Output the product of the multiplication of two floating point input images
 * @ingroup imageMatrix

 * @param Pointer to output floating point encoded image
 * @param Pointer to multiplication source floating point encoded image
 * @param Pointer to multiplication source floating point encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images

 * @retval 1=success,0=error
 */
int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height, unsigned int channels );



/**
 * @brief Output the product of the multiplication of two floating point input images
 * @ingroup imageMatrix

 * @param Pointer to output floating point encoded image
 * @param Pointer to multiplication source floating point encoded image
 * @param Pointer to multiplication source unsigned char encoded image
 * @param width of source and target images
 * @param height of source and target  images
 * @param channels of source and target images

 * @retval 1=success,0=error
 */
int multiply2DMatricesFWithUC(float * out , float * mult1 , unsigned char * mult2 , unsigned int width , unsigned int height, unsigned int channels );


#endif // IMAGEMATRIX_H_INCLUDED
