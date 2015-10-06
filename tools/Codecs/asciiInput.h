#ifndef _ASCIIINPUT_H_INCLUDED
#define _ASCIIINPUT_H_INCLUDED

#include "codecs.h"


int ReadASCII(char * filename,struct Image * pic,char read_only_header);
int WriteASCII(char * filename,struct Image * pic,int packed);


#endif
