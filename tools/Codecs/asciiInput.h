#ifndef _PPMINPUT_H_INCLUDED
#define _PPMINPUT_H_INCLUDED

#include "codecs.h"

int ReadPPM(char * filename,struct Image * pic,char read_only_header);
int WritePPM(char * filename,struct Image * pic);


#endif
