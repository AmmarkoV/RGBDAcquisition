#ifndef _PNGINPUT_H_INCLUDED
#define _PNGINPUT_H_INCLUDED

#include "codecs.h"

#if USE_PNG_FILES

int ReadPNG( char *filename,struct Image * pic,char read_only_header);
int WritePNG(char * filename,struct Image * pic);

#endif

#endif // IMAGE_STORAGE_PNG_H_INCLUDED
