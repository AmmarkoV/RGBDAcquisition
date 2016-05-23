#include "reconstruction.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int  reconstruct3D(const char * filenameLeft )
{
  fprintf(stderr,"reconstruct3D(%s)\n",filenameLeft);

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filenameLeft,"r");
  if (fp!=0)
  {

    char * line = NULL;
    char * lineStart = line;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        lineStart = line;
        while (*lineStart==' ') { ++lineStart; }

        printf("Retrieved line of length %zu :\n", read);
        printf("%s", lineStart);
        char * varNameEnd = strchr(lineStart , ' ');
        if (varNameEnd!=0)
        {
         *varNameEnd=0;
         printf("VAR = %s\n", lineStart);
         char * val = varNameEnd+1;
         printf("VAL = %s\n", val);

        }
    }

    fclose(fp);
    if (line) { free(line); }
    return 1;
}

fprintf(stderr,"Done.. \n");
return 0;
}
