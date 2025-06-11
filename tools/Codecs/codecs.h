/**
   @file codecs.h
*  @brief The main Acquisition library that handles plugins and provides .
**************************************************************************
* Copyright (C) 2010 by Ammar Qammaz *
* ammarkov@gmail.com *
* *
* This program is free software; you can redistribute it and/or modify *
* it under the terms of the GNU General Public License as published by *
* the Free Software Foundation; either version 2 of the License, or *
* (at your option) any later version. *
* *
* This program is distributed in the hope that it will be useful, *
* but WITHOUT ANY WARRANTY; without even the implied warranty of *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the *
* GNU General Public License for more details. *
* *
* You should have received a copy of the GNU General Public License *
* along with this program; if not, write to the *
* Free Software Foundation, Inc., *
* 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. *
***************************************************************************/

#ifndef CODECS_H_INCLUDED
#define CODECS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


#include "../Primitives/image.h"

//FORCE SOME LIBRARIES HERE MAYBE ( this breaks the CMake Way of enabling / disabling libs )
//#define USE_JPG_FILES 1
//#define USE_PNG_FILES 1
#define USE_PFM_FILES 1
#define USE_PPM_FILES 1
#define USE_BMP_FILES 1
#define USE_ASCII_FILES 1


#define READ_CREATES_A_NEW_PIXEL_BUFFER 1

enum codecTypeList
{
   NO_CODEC = 0,
   JPG_CODEC ,
   PFM_CODEC ,
   PPM_CODEC ,
   COMPATIBLE_PNM_CODEC ,
   PNM_CODEC ,
   PNG_CODEC ,
   ASCII_CODEC ,
   BMP_CODEC ,
   //- - - - - - - - -
   EXISTING_CODECS
};

/*
struct Image
{
  unsigned char * pixels;
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  unsigned int image_size;
  unsigned int timestamp;
};
*/


unsigned int simplePowCodecs(unsigned int base,unsigned int exp);

int refreshImage(struct Image * img);

struct Image * readImage( char *filename,unsigned int type,char read_only_header);
unsigned char * readImageRaw( char *filename,unsigned int type,unsigned int *width,unsigned int *height,unsigned int *bitsperpixel , unsigned int *channels);

int writeImageFile(struct Image * pic,unsigned int type,char *filename);
int writeImageMemory(struct Image * pic,unsigned int type,char *mem,unsigned long * mem_size);

int populateImage(struct Image * img , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel, unsigned char * pixels);

struct Image * createImageUsingExistingBuffer( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel , unsigned char * pixels);
struct Image * createImage( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel);




/**
 * @brief Destroy loaded image
 * @ingroup codecs
 * @param Image structure pointer to be destroyed
 * @retval 1=Success,0=Failure
 */
int destroyImage(struct Image * img);


struct Image * createSameDimensionsImage( struct Image * inputImage);



/**
 * @brief Copy Image and return a newly allocated Image Structure
 * @ingroup codecs
 * @param Image structure pointer to be copied
 * @retval Pointer to Image Structure ,0=Failure
 */
struct Image * copyImage( struct Image * inputImage);


struct Image * createImageBitBlt( struct Image * inImg , unsigned int x , unsigned int y , unsigned int width , unsigned int height );

int swapImageEndianness(struct Image * img);
int swapImageEndiannessRaw(unsigned char * pixels, unsigned int width,unsigned int height,unsigned int bitsperpixel , unsigned int channels);


int convertCodecImages(char * filenameInput , char * filenameOutput);


unsigned int guessFilenameTypeStupid(char * filename);
#ifdef __cplusplus
}
#endif


#endif // IMAGE_STORAGE_H_INCLUDED
