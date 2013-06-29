#ifndef _JPGINPUT_H_INCLUDED
#define _JPGINPUT_H_INCLUDED


#include "codecs.h"

#define USE_JPG_FILES 1
#define USE_PNG_FILES 1
#define USE_PNM_FILES 1

int ReadJPEG( char *filename,struct Image * pic,char read_only_header);

int WriteJPEGFile(struct Image * pic,char *filename);
int WriteJPEGMemory(struct Image * pic,char *mem,unsigned long * mem_size);


#endif // _JPG_H_INCLUDED
