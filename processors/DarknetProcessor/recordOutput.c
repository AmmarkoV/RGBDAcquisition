#include "recordOutput.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char loggingDir[1024]={0};


int resumeFrameOutput()
{
   time_t clock = time(NULL);
   struct tm * ptm = gmtime ( &clock );
   char commandToRun[1024];
   snprintf(commandToRun,1024,"cat %u_%02u_%02u/surveilance.log | tail -n 1  | cut -d ',' -f7 ", EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday);

   /* Open the command for reading. */
   FILE * fp = popen(commandToRun, "r");
   if (fp == 0 )
        {
           fprintf(stderr,"Failed to run command (%s) \n",commandToRun);
           return 0;
         }

   /* Read the output a line at a time - output it. */
   char what2GetBack[512];
   unsigned int what2GetBackMaxSize=512;
   unsigned int lineNumber=1;

   unsigned int i=0;
   while (fgets(what2GetBack, what2GetBackMaxSize , fp) != 0)
    {
        ++i;
        if (lineNumber==i) { break; }
    }
  /* close */
  pclose(fp);

  return atoi(what2GetBack);
}

int makeDirectory(const char * path)
{
  char command[1024]={0};
  snprintf(command,1024,"mkdir -p %s",path);
  int i=system(command);
  return (i==0);
}



int useLoggingDirectory(const char * path)
{
  if (strcmp(path,loggingDir)!=0)
  {
    if ( makeDirectory(path) )
      {
        snprintf(loggingDir,1024,"%s",path);
      }
  }
  return 1;
}


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
            fp, "detection(%u,%u,%u,%u,%u,%u,%u,%s,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f)\n",

            //Year/Month/Day
            EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year,
            ptm->tm_mon+1,
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
     return 1;
   } else
   { fprintf(stderr,"Unable to log..\n"); }

 return 0;
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
