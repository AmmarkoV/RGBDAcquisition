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

#include "jpgInput.h"
#include "pngInput.h"
#include "ppmInput.h"



#define DEBUG_READING_IMAGES 0

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
       if (!ReadJPEG(filename,img,read_only_header)) { free(img); img=0; }
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
       if (!ReadPNG(filename,img,read_only_header)) { free(img); img=0; }
        #if DEBUG_READING_IMAGES
	     char ppmfilename[512]={0};
	     strcpy(ppmfilename,filename);
	     strcat(ppmfilename,".ppm");
	     WritePPM(ppmfilename,img);
	    #endif
      break;
     #endif

       case PPM_CODEC :
       if (!ReadPPM(filename,img,read_only_header)) { free(img); img=0; }
       break;

      default :
       free(img);
       img=0;
      break;
   };

   return img;
}


int writeImageFile(struct Image * pic,unsigned int type,char *filename)
{
   switch (type)
   {
     #if USE_JPG_FILES
      case JPG_CODEC :
       return WriteJPEGFile(pic,filename);
     #endif // USE_JPG_FILES


      case PPM_CODEC :
      case PNM_CODEC :
       WritePPM(filename,pic);
      break;



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


int destroyImage(struct Image * img)
{
    if (img->pixels!=0) { free(img->pixels); img->pixels=0; }
    if (img!=0) { free(img); img=0; }
    return 0;
}
