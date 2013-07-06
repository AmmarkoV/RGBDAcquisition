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

#ifndef CODECS_H_INCLUDED
#define CODECS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


#define USE_JPG_FILES 1
#define USE_PNG_FILES 1


#define READ_CREATES_A_NEW_PIXEL_BUFFER 1

enum codecTypeList
{
   NO_CODEC = 0,
   JPG_CODEC ,
   PPM_CODEC ,
   PNM_CODEC ,
   PNG_CODEC
};

struct Image
{
  unsigned char * pixels;
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  unsigned int image_size;
};

struct Image * readImage( char *filename,unsigned int type,char read_only_header);

int writeImageFile(struct Image * pic,unsigned int type,char *filename);
int writeImageMemory(struct Image * pic,unsigned int type,char *mem,unsigned long * mem_size);

int destroyImage(struct Image * img);

#ifdef __cplusplus
}
#endif


#endif // IMAGE_STORAGE_H_INCLUDED
