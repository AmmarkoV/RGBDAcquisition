#ifndef RECORDOUTPUT_H_INCLUDED
#define RECORDOUTPUT_H_INCLUDED

#include <stdio.h>
#include <time.h>



#define EPOCH_YEAR_IN_TM_YEAR 1900

int makeDirectory(const char * path);
int useLoggingDirectory(const char * path);

int resumeFrameOutput();

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
