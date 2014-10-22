/***************************************************************************
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

#include "codecs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if USE_JPG_FILES
      #include "jpgInput.h"
#else
  #warning "JPG Support is disabled in this build of Image Codecs"
#endif // USE_JPG_FILES


#if USE_PNG_FILES
      #include "pngInput.h"
#else
  #warning "PNG Support is disabled in this build of Image Codecs"
#endif // USE_PNG_FILES


#if USE_PPM_FILES
      #include "ppmInput.h"
#else
  #error "PNM/PPM Support is disabled in this build of Image Codecs and this doesnt make any sense since we have it hardcoded"
#endif // USE_PPM_FILES

#define DEBUG_READING_IMAGES 0



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


unsigned int simplePow(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}



struct Image * readImage( char *filename,unsigned int type,char read_only_header)
{

   struct Image * img = 0;
   img = (struct Image *) malloc( sizeof(struct Image) );
   memset(img,0,sizeof(struct Image));
   img->pixels=0; // :P just to make sure

   switch (type)
   {
     #if USE_JPG_FILES
      case JPG_CODEC :
       if (!ReadJPEG(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using jpg reader" NORMAL , filename);
           free(img);
           img=0;
         }
        #if DEBUG_READING_IMAGES
	     char ppmfilename[512]={0};
	     strcpy(ppmfilename,filename);
	     strcat(ppmfilename,".ppm");
	     WritePPM(ppmfilename,img);
	    #endif
      break;
     #endif

     #if USE_PNG_FILES
      case PNG_CODEC :
       if (!ReadPNG(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using png reader" NORMAL , filename);
           free(img);
           img=0;
         }
        #if DEBUG_READING_IMAGES
	     char ppmfilename[512]={0};
	     strcpy(ppmfilename,filename);
	     strcat(ppmfilename,".ppm");
	     WritePPM(ppmfilename,img);
	    #endif
      break;
     #endif

     #if USE_PPM_FILES
       case PPM_CODEC :
       case PNM_CODEC :
       if (!ReadPPM(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using pnm reader" NORMAL , filename);
           free(img);
           img=0;
         }
       break;
     #endif

      default :
       free(img);
       img=0;
      break;
   };

   return img;
}


unsigned char * readImageRaw( char *filename,unsigned int type,unsigned int *width,unsigned int *height,unsigned int *bitsperpixel , unsigned int *channels)
{
    struct Image * img = readImage(filename,type,0);
    if (img!=0)
    {
        unsigned char * pixels = img->pixels;
        *width=img->width;
        *height=img->height;
        *bitsperpixel=img->bitsperpixel;
        *channels=img->channels;
        //fprintf(stderr,"Read %s of %ux%u %u bpp %u channels\n",filename,img->width,img->height,img->bitsperpixel,img->channels);

        free(img);
        return pixels;
    }
   return 0;
}




int swapImageEndianness(struct Image * img)
{
  unsigned char * traverser=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap1=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap2=(unsigned char * ) img->pixels;

  unsigned int bytesperpixel = (img->bitsperpixel/8);
  unsigned char * endOfMem = traverser + img->width * img->height * img->channels * bytesperpixel;

  unsigned char tmp ;
  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    tmp = *traverserSwap1;
    *traverserSwap1 = *traverserSwap2;
    *traverserSwap2 = tmp;

    traverser += bytesperpixel;
  }

 return 1;
}

int swapImageEndiannessRaw(unsigned char * pixels, unsigned int width,unsigned int height,unsigned int bitsperpixel , unsigned int channels)
{
  struct Image imgS={0};

  imgS.bitsperpixel=bitsperpixel;
  imgS.channels=channels;
  imgS.width=width;
  imgS.height=height;
  imgS.pixels=pixels;

  return swapImageEndianness(&imgS);
}








int writeImageFile(struct Image * pic,unsigned int type,char *filename)
{
   switch (type)
   {
     #if USE_JPG_FILES
      case JPG_CODEC :
       return WriteJPEGFile(pic,filename);
     #endif // USE_JPG_FILES


     #if USE_PPM_FILES
      case PPM_CODEC :
      case PNM_CODEC :
       WritePPM(filename,pic);
      break;
    #endif


     #if USE_PNG_FILES
      case PNG_CODEC :
       if (!WritePNG(filename,pic)) { free(pic); pic=0; }
        #if DEBUG_READING_IMAGES
	     char ppmfilename[512]={0};
	     strcpy(ppmfilename,filename);
	     strcat(ppmfilename,".png");
	     WritePNG(ppmfilename,pic);
	    #endif
      break;
     #endif

      default :
        break;
   };

   return 0;
}


int writeImageMemory(struct Image * pic,unsigned int type,char *mem,unsigned long * mem_size)
{
   switch (type)
   {
     #if USE_JPG_FILES
      case JPG_CODEC :
       return WriteJPEGMemory(pic,mem,mem_size);
     #endif // USE_JPG_FILES

      default :
        break;
   };

   return 0;
}


struct Image * createImageUsingExistingBuffer( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel , unsigned char * pixels)
{
  struct Image * img = 0;
  img = (struct Image *) malloc( sizeof(struct Image) );
  if (img == 0 ) { fprintf(stderr,"Could not allocate a new image %ux%u %u channels %u bitsperpixel\n",width,height,channels,bitsPerPixel); return 0; }
  memset(img,0,sizeof(struct Image));

  img->width = width;
  img->height = height;
  img->channels = channels;
  img->bitsperpixel = bitsPerPixel;

  img->pixels = pixels;
  return  img;
}


struct Image * createImage( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel)
{
  unsigned char * pixels = ( unsigned char * ) malloc(width * height * channels * (bitsPerPixel/8) * sizeof(unsigned char) );
  if (pixels==0) { fprintf(stderr,"Could not allocate a new %ux%u image \n",width,height); return 0; }
  memset(pixels,0,width * height * channels * (bitsPerPixel/8) * sizeof(unsigned char));

  struct Image * img =  createImageUsingExistingBuffer(width , height , channels , bitsPerPixel , pixels);
  if (img==0) { free(pixels); return 0; }

  return  img;
}


int destroyImage(struct Image * img)
{
    if (img==0) {return 0; }
    if (img->pixels!=0) { free(img->pixels); img->pixels=0; }
    if (img!=0) { free(img); img=0; }
    return 1;
}


