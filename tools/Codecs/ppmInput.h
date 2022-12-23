#ifndef _PPMINPUT_H_INCLUDED
#define _PPMINPUT_H_INCLUDED

#include "codecs.h"

int ReadPPM(const char * filename,struct Image * pic,char read_only_header);
int ReadSwappedPPM(const char * filename,struct Image * pic,char read_only_header);


int WritePPM(const char * filename,struct Image * pic);
int WriteSwappedPPM(const char * filename,struct Image * pic);

#endif
