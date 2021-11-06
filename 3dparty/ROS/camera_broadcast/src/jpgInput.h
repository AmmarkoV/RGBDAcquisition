#ifndef _JPGINPUT_H_INCLUDED
#define _JPGINPUT_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


#include "image.h"

#define USE_JPG_FILES 1

int ReadJPEG( char *filename,struct Image * pic,char read_only_header);

int WriteJPEGFile(struct Image * pic,char *filename);
int WriteJPEGMemory(struct Image * pic,char *mem,unsigned long * mem_size,int quality);

#ifdef __cplusplus
}
#endif


#endif // _JPG_H_INCLUDED
