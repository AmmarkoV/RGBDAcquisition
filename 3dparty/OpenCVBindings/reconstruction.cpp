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


        char * num1 = lineStart; // number1 start to first ' '
        char * num2 = strchr(num1 , ' ');
        if (num2!=0) { *num2=0; ++num2; } else { fprintf(stderr,"oops"); }
        char * num3 = strchr(num2 , ' ');
        if (num3!=0) { *num3=0; ++num3; } else { fprintf(stderr,"oops"); }
        char * num4 = strchr(num3, ' ');

        printf("vals are |%s|%s|%s|%s| \n", num1,num2,num3,num4);
        printf("floats are |%0.2f|%0.2f|%0.2f|%0.2f| \n",atof(num1),atof(num2),atof(num3),atof(num4));

    }

    fclose(fp);
    if (line) { free(line); }
    return 1;
}

fprintf(stderr,"Done.. \n");
return 0;
}
