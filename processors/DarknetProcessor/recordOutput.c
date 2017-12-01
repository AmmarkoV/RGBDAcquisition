#include "recordOutput.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPOCH_YEAR_IN_TM_YEAR 1900

FILE * startLogging(const char * filename)
{
  return fopen(filename,"a");
}

int logEvent(
              FILE * fp,
              struct tm * ptm,
              unsigned int frameNumber,
              float x,
              float y,
              float width,
              float height,
              const char * label,
              float probability
            )
{
  if (fp!=0)
  {
   if (strcmp(label,"person")==0)
   {
    fprintf(
            fp,
            "detection(%u,%u,%u,%u,%u,%u,%u,%s,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f)\n",

            //Year/Month/Day
            EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year,
            ptm->tm_mon,
            ptm->tm_mday,

            //Hour/Min/Sec
            ptm->tm_hour,
            ptm->tm_min,
            ptm->tm_sec,

            frameNumber,
            label,
            probability,
            x,
            y,
            width,
            height
           );
    }
   }
   return 1;
}


int stopLogging(FILE * fp)
{
  if (fp!=0)
    {
      fclose(fp);
      return 1;
    }
  return 0;
}
