#include "jpgExifexternal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//This is a more complex check , could use it in the future
int scanJPGFileForExifTags(char * filename)
{
    char command[1024]={0};
    strcpy(command,"exif \""); strncat(command,filename,1000); strcat(command,"\"");

   FILE *fp=0;
    /* Open the command for reading. */
     fp = popen(command, "r");
     if (fp == 0 )
       {
         fprintf(stderr,"Failed to run command\n");
         return 0;
       }

 /* Read the output a line at a time - output it. */
  char output[2048]={0};
  unsigned int size_of_output = 2048;

  unsigned int i=0;
  while (fgets(output, size_of_output , fp) != 0)
    {
        ++i;
         fprintf(stderr,"\n\nline %u = %s \n",i,output);
        break;
    }

  /* close */
  pclose(fp);
  return (i>0);
}
