#ifndef RECORDOUTPUT_H_INCLUDED
#define RECORDOUTPUT_H_INCLUDED

#include <stdio.h>
#include <time.h>

FILE * startLogging(const char * filename);

int logEvent(
              FILE * fp ,
              struct tm * ptm,
              unsigned int frameNumber,
              float x,
              float y,
              float width,
              float height,
              const char * label,
              float probability
            );
int stopLogging(FILE * fp);

#endif // RECORDOUTPUT_H_INCLUDED
