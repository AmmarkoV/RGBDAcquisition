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
 // #warning "JPG Support active"
#else
  #warning "JPG Support is disabled in this build of Image Codecs"
#endif // USE_JPG_FILES


#if USE_PNG_FILES
      #include "pngInput.h"
  // #warning "PNG Support active"
#else
  #warning "PNG Support is disabled in this build of Image Codecs"
#endif // USE_PNG_FILES


#if USE_BMP_FILES
      #include "bmpInput.h"
  // #warning "BMP Support active"
#else
  #warning "BMP Support is disabled in this build of Image Codecs"
#endif // USE_BMP_FILES



#if USE_PPM_FILES
      #include "ppmInput.h"
#else
  #error "PNM/PPM Support is disabled in this build of Image Codecs and this doesnt make any sense since we have it hardcoded"
#endif // USE_PPM_FILES



#if USE_PFM_FILES
      #include "pfmInput.h"
#else
  #error "PFM Support is disabled in this build of Image Codecs and this doesnt make any sense since we have it hardcoded"
#endif // USE_PPM_FILES




#if USE_ASCII_FILES
      #include "asciiInput.h"
#else
  #error "ASCII Support is disabled in this build of Image Codecs and this doesnt make any sense since we have it hardcoded"
#endif // USE_ASCII_FILES


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


unsigned int simplePowCodecs(unsigned int base,unsigned int exp)
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


int refreshImage(struct Image * img)
{
  if (img==0)            { return 0; }
  if (img->pixels == 0 ) { fprintf(stderr,"Image has no allocated pixel buffer \n"); return 0; }

  unsigned int logicalSize = img->width * img->height * img->channels * (img->bitsperpixel /8);
  if (img->image_size != logicalSize )
    {
      fprintf(stderr,"Image has an inconsistent size, adjusting it\n");
      img->image_size = logicalSize;
    }

 return 1;
}


unsigned int guessFilenameTypeStupid(char * filename)
{
  fprintf(stderr,"Guessing filename type for `%s` \n" , filename);
  if (strcasestr(filename,".BMP")!=0)   { return BMP_CODEC; }            else
  if (strcasestr(filename,".RLE")!=0)   { return BMP_CODEC; }            else
  if (strcasestr(filename,".JPG")!=0)   { return JPG_CODEC; }            else
  if (strcasestr(filename,".JPEG")!=0)  { return JPG_CODEC; }            else
  if (strcasestr(filename,".PNG")!=0)   { return PNG_CODEC; }            else
  if (strcasestr(filename,".CPNM")!=0)  { return COMPATIBLE_PNM_CODEC; } else
  if (strcasestr(filename,".PNM")!=0)   { return PNM_CODEC; }            else
  if (strcasestr(filename,".PPM")!=0)   { return PPM_CODEC; }            else
  if (strcasestr(filename,".ASCII")!=0) { return ASCII_CODEC; }          else
  if (strcasestr(filename,".TEXT")!=0)  { return ASCII_CODEC; }          else
  if (strcasestr(filename,".TXT")!=0)   { return ASCII_CODEC; }

 return  NO_CODEC;
}



struct Image * readImage( char *filename,unsigned int type,char read_only_header)
{

   struct Image * img = 0;
   img = (struct Image *) malloc( sizeof(struct Image) );
   memset(img,0,sizeof(struct Image));
   img->pixels=0; // :P just to make sure


   if (type==NO_CODEC)
   {
     unsigned int suggestedType = guessFilenameTypeStupid(filename);
     type=suggestedType;
     fprintf(stderr,"Using type %u loader for image %s \n",type,filename);
   }


   switch (type)
   {
     case JPG_CODEC :
     #if USE_JPG_FILES
       //fprintf(stderr,GREEN "JPG Loader active" NORMAL , filename);
       if (!ReadJPEG(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using jpg reader" NORMAL , filename);
           free(img);
           img=0;
         }
        #if DEBUG_READING_IMAGES
         char ppmfilename[513]={0};
         snprintf(ppmfilename,512,"%s.ppm",filename);
         //strcpy(ppmfilename,filename);
         //strcat(ppmfilename,".ppm");
         WritePPM(ppmfilename,img);
        #endif
     #else
       fprintf(stderr,RED "JPG File requested (%s) , but this build of Codec Library does not have JPG Support :(" NORMAL , filename);
     #endif
     break;

     case PFM_CODEC :
     #if USE_PFM_FILES
       //fprintf(stderr,GREEN "PNG Loader active" NORMAL , filename);
       if (!ReadPFM(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using pfm reader" NORMAL , filename);
           free(img);
           img=0;
         }
        #if DEBUG_READING_IMAGES
         char ppmfilename[513]={0};
         snprintf(ppmfilename,512,"%s.ppm",filename);
         //strcpy(ppmfilename,filename);
         //strcat(ppmfilename,".ppm");
         WritePPM(ppmfilename,img);
       #endif
     #else
       fprintf(stderr,RED "PFM File requested (%s) , but this build of Codec Library does not have PFM Support :(" NORMAL , filename);
     #endif
     break;

     case PNG_CODEC :
     #if USE_PNG_FILES
       //fprintf(stderr,GREEN "PNG Loader active" NORMAL , filename);
       if (!ReadPNG(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Could error reading file %s using png reader" NORMAL , filename);
           free(img);
           img=0;
         }
        #if DEBUG_READING_IMAGES
         char ppmfilename[513]={0};
         snprintf(ppmfilename,512,"%s.ppm",filename);
         //strcpy(ppmfilename,filename);
         //strcat(ppmfilename,".ppm");
         WritePPM(ppmfilename,img);
        #endif
     #else
       fprintf(stderr,RED "PNG File requested (%s) , but this build of Codec Library does not have PNG Support :(" NORMAL , filename);
     #endif
     break;



      case COMPATIBLE_PNM_CODEC :
       #if USE_PPM_FILES
        if (!ReadSwappedPPM(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Error reading file %s using swapped pnm reader" NORMAL , filename);
           free(img);
           img=0;
         }
       #else
         fprintf(stderr,RED "Swapped PNM/PPM File requested (%s) , but this build of Codec Library does not have PNM/PPM Support :(" NORMAL , filename);
       #endif
      break;


     case PPM_CODEC :
     case PNM_CODEC :
     #if USE_PPM_FILES
       if (!ReadPPM(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Error reading file %s using pnm reader" NORMAL , filename);
           free(img);
           img=0;
         }
     #else
       fprintf(stderr,RED "PNM/PPM File requested (%s) , but this build of Codec Library does not have PNM/PPM Support :(" NORMAL , filename);
     #endif
     break;


     case ASCII_CODEC :
     #if USE_ASCII_FILES
        if (!ReadASCII(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Error reading file %s using pnm reader" NORMAL , filename);
           free(img);
           img=0;
         }
     #else
       fprintf(stderr,RED "PNM/PPM File requested (%s) , but this build of Codec Library does not have PNM/PPM Support :(" NORMAL , filename);
     #endif // USE_ASCII_FILES
     break;


     case BMP_CODEC :
     #if USE_BMP_FILES
       if (!ReadBMP(filename,img,read_only_header))
         {
           fprintf(stderr,RED "Error reading file %s using bmp reader" NORMAL , filename);
           free(img);
           img=0;
         }
     #else
      fprintf(stderr,RED "BMP File requested (%s) , but this build of Codec Library does not have PNM/PPM Support :(" NORMAL , filename);
     #endif
     break;


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
  if (img==0) { return 0; }
  if (img->pixels==0) { return 0; }
  unsigned char * traverser=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap1;//=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap2;//=(unsigned char * ) img->pixels;

  unsigned int bytesperpixel = (img->bitsperpixel/8);
  unsigned char * endOfMem = traverser + img->width * img->height * img->channels * bytesperpixel;

  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    unsigned char tmp = *traverserSwap1;
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
 if (filename==0) {return 0; }
 if (pic==0) {return 0; }
 if (pic->pixels==0) {  return 0; }


   switch (type)
   {
     #if USE_JPG_FILES
      case JPG_CODEC :
       return WriteJPEGFile(pic,filename);
     #else
      case JPG_CODEC :
      fprintf(stderr,"JPG file writing is not compiled in this build .. \n");
      break;
     #endif // USE_JPG_FILES


     #if USE_PPM_FILES
      case COMPATIBLE_PNM_CODEC :
       WriteSwappedPPM(filename,pic);
      break;

      case PPM_CODEC :
      case PNM_CODEC :
       WritePPM(filename,pic);
      break;
    #else
      case COMPATIBLE_PNM_CODEC :
      case PPM_CODEC :
      case PNM_CODEC :
      fprintf(stderr,"PPM file writing is not compiled in this build .. \n");
      break;
    #endif


     #if USE_PNG_FILES
      case PNG_CODEC :
       if (!WritePNG(filename,pic)) { free(pic); pic=0; }
        #if DEBUG_READING_IMAGES
         char ppmfilename[513]={0};
         snprintf(ppmfilename,512,"%s.png",filename);
         //strcpy(ppmfilename,filename);
         //strcat(ppmfilename,".png");
         WritePNG(ppmfilename,pic);
        #endif
      break;
     #else
      case PNG_CODEC :
      fprintf(stderr,"PNG file writing is not compiled in this build .. \n");
      break;
     #endif



     #if USE_ASCII_FILES
     case ASCII_CODEC :
         WriteASCII(filename,pic,0);
      break;
     #else
      case ASCII_CODEC :
      fprintf(stderr,"ASCII file writing is not compiled in this build .. \n");
      break;
     #endif // USE_ASCII_FILES



     #if USE_BMP_FILES
     case BMP_CODEC :
         WriteBMP(filename,pic);
      break;
     #else
      case BMP_CODEC :
      fprintf(stderr,"BMP file writing is not compiled in this build .. \n");
      break;
     #endif // USE_ASCII_FILES

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
       return WriteJPEGMemory(pic,mem,mem_size,75);
     #endif // USE_JPG_FILES

      default :
       fprintf(stderr,"file writing in memory is not availiable for this file type.. \n");
      break;
   };

   return 0;
}




int populateImage(struct Image * img , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel, unsigned char * pixels)
{
  if (img == 0 ) { fprintf(stderr,"Could not populateImage empty image\n"); return 0; }
  memset(img,0,sizeof(struct Image));

  img->width = width;
  img->height = height;
  img->channels = channels;
  img->bitsperpixel = bitsPerPixel;

  img->pixels = pixels;
  return  1;
}




struct Image * createImageUsingExistingBuffer( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel , unsigned char * pixels)
{
  struct Image * img = 0;
  img = (struct Image *) malloc( sizeof(struct Image) );
  if (img == 0 ) { fprintf(stderr,"Could not allocate a new image %ux%u %u channels %u bitsperpixel\n",width,height,channels,bitsPerPixel); return 0; }


  if ( populateImage(img,width,height,channels,bitsPerPixel,pixels) )
  {
    return img;
  }

  free(img);
  return  0;
}




struct Image * createImage( unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsPerPixel)
{
  unsigned char * pixels = ( unsigned char * ) malloc(width * height * channels * (bitsPerPixel/8) * sizeof(unsigned char) );
  if (pixels==0) { fprintf(stderr,"Could not allocate a new %ux%u image \n",width,height); return 0; }

  //We are very clean and we clear memory..
  memset(pixels,0,width * height * channels * (bitsPerPixel/8) * sizeof(unsigned char));

  struct Image * img =  createImageUsingExistingBuffer(width , height , channels , bitsPerPixel , pixels);
  if (img==0) { free(pixels); return 0; }

  return  img;
}





#define MEMPLACE1(x,y,width) ( y * ( width  ) + x )
#define MEMPLACE3(x,y,width) ( ( y * ( width * 3 ) ) + (x*3) )


int bitBltCodecCopy(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
                            unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                            unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }
  if ( (sourceWidth==0)&&(sourceHeight==0) ) { return 0; }

  fprintf(stderr,"BitBlt an area of target image %u,%u  sized %u,%u \n",tX,tY,targetWidth,targetHeight);
  fprintf(stderr,"BitBlt an area of source image %u,%u  sized %u,%u \n",sX,sY,sourceWidth,sourceHeight);
  fprintf(stderr,"BitBlt size was width %u height %u \n",width,height);
  //Check for bounds -----------------------------------------
  unsigned int boundsCheckFired=0;
  if (tX+width>=targetWidth) { width=targetWidth-tX;   boundsCheckFired+=1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  boundsCheckFired+=10;  }

  if (sX+width>=sourceWidth) { width=sourceWidth-sX;  boundsCheckFired+=100; }
  if (sY+height>=sourceHeight) { height=sourceHeight-sY-1; boundsCheckFired+=1000; }
  //----------------------------------------------------------
  if (boundsCheckFired) { fprintf(stderr,"Bounds Check fired %u ..!\n",boundsCheckFired) ;}
  fprintf(stderr,"BitBlt size NOW is width %u height %u \n",width,height);
  if ( (width==0) || (height==0) ) { fprintf(stderr,"Unacceptable size..\n"); return 0;}
  if ( (width>sourceWidth) || (width>targetWidth) ) { fprintf(stderr,"Unacceptable size , too big..\n"); return 0;}
  if ( (height>sourceHeight) || (height>targetHeight) ) { fprintf(stderr,"Unacceptable size , too big..\n"); return 0;}

  unsigned char *  sourcePTR      = source+ MEMPLACE3(sX,sY,sourceWidth);
  unsigned char *  sourceLimitPTR = source+ MEMPLACE3((sX+width),(sY+height),sourceWidth);
  unsigned int     sourceLineSkip = (sourceWidth-width) * 3;
  unsigned char *  sourceLineLimitPTR = sourcePTR + (width*3) ; /*-3 is required here*/
  //fprintf(stderr,"SOURCE (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX+width,sY+height);
  //fprintf(stderr,"sourcePTR is %p , limit is %p \n",sourcePTR,sourceLimitPTR);
  //fprintf(stderr,"sourceLineSkip is %u\n",        sourceLineSkip);
  //fprintf(stderr,"sourceLineLimitPTR is %p\n",sourceLineLimitPTR);


  unsigned char * targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  unsigned char * targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  unsigned int targetLineSkip = (targetWidth-width) * 3;
  unsigned char * targetLineLimitPTR = targetPTR + (width*3) ; /*-3 is required here*/
  //fprintf(stderr,"TARGET (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX+width,tY+height);
  //fprintf(stderr,"targetPTR is %p , limit is %p \n",targetPTR,targetLimitPTR);
  //fprintf(stderr,"targetLineSkip is %u\n", targetLineSkip);
  //fprintf(stderr,"targetLineLimitPTR is %p\n",targetLineLimitPTR);

  while ( (sourcePTR < sourceLimitPTR) && ( targetPTR < targetLimitPTR ) )
  {
     while ( (sourcePTR < sourceLineLimitPTR) && ((targetPTR < targetLineLimitPTR)) )
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
     }

    sourceLineLimitPTR += sourceWidth*3;
    targetLineLimitPTR += targetWidth*3;
    sourcePTR+=sourceLineSkip;
    targetPTR+=targetLineSkip;
  }

 return 1;
}





struct Image * createImageBitBlt( struct Image * inImg , unsigned int x , unsigned int y , unsigned int width , unsigned int height )
{
  struct Image * img = createImage( width , height , inImg->channels , inImg->bitsperpixel);

  //Bit BLT will replace all pixels so this is not needed ,this is a test to make sure it covers all the area..
  //memset(img->pixels,0,width * height * inImg->channels * (inImg->bitsperpixel/8) * sizeof(unsigned char));

  bitBltCodecCopy(img->pixels, 0 ,  0 , width , height ,
                  inImg->pixels , x, y , inImg->width , inImg->height ,
                  width , height);


  return  img;
}


int destroyImage(struct Image * img)
{
    if (img==0) {return 0; }
    if (img->pixels!=0) { free(img->pixels); img->pixels=0; }
    if (img!=0) { free(img); /*img=0;*/ }
    return 1;
}




struct Image * createSameDimensionsImage( struct Image * inputImage)
{
  if (inputImage==0) { fprintf(stderr,"Could not copy null image\n"); return 0; }
  if (inputImage->pixels==0) { fprintf(stderr,"Could not copy null image buffer\n"); return 0; }

 struct Image * newImage = createImage(inputImage->width,inputImage->height,inputImage->channels,inputImage->bitsperpixel );

 return newImage;
}


struct Image * copyImage( struct Image * inputImage)
{
  if (inputImage==0) { fprintf(stderr,"Could not copy null image\n"); return 0; }
  if (inputImage->pixels==0) { fprintf(stderr,"Could not copy null image buffer\n"); return 0; }

 struct Image * newImage = createImage(inputImage->width,inputImage->height,inputImage->channels,inputImage->bitsperpixel );
 unsigned int logicalSize = inputImage->width * inputImage->height * inputImage->channels * (inputImage->bitsperpixel /8);

 if (logicalSize!=inputImage->image_size)
 {
   fprintf(stderr,"copyImage detected an initially bigger inputImage , will use current size.. \n");
 }
 memcpy(newImage->pixels,inputImage->pixels,logicalSize);

 return newImage;
}



int convertCodecImages(char * filenameInput , char * filenameOutput)
{
 unsigned int inputType = guessFilenameTypeStupid(filenameInput);
 struct Image * inputImage = readImage(filenameInput,inputType,0);
 if (inputImage!=0)
 {
    unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
    writeImageFile(inputImage,outputType ,filenameOutput);

    destroyImage(inputImage);
    return 1;
 }
 return 0;
}
