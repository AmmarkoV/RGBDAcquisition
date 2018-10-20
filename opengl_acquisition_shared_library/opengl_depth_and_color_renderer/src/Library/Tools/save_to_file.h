/** @file save_to_file.h
 *  @brief  Code to save images from memory to files using PNM/PPM File format
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef SAVE_TO_FILE_H_INCLUDED
#define SAVE_TO_FILE_H_INCLUDED


/**
* @brief Save Image from a pixel buffer to a file
* @ingroup OGLRendererSandbox
* @param filename , A string with the filename of output file
* @param pixels , A memory pointer to the frame we want to save
* @param width, Width of the buffer
* @param height , Height of the buffer
* @param channels , Number of color channels for the buffer ( RGB =  3 , GrayScale = 1 )
* @param channels , Bits per channel ( RGB = 8 ( 3channels * 8 Bits ) , Depth Grayscale = 16 ( 1 channel * 16 bits )  )
* @retval 1=Success , 0=Failure
*/
int saveRawImageToFileOGLR(char * filename,void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperchannel);

#endif // SAVE_TO_FILE_H_INCLUDED
