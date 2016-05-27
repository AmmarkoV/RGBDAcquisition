#ifndef PRIMITIVES_H_INCLUDED
#define PRIMITIVES_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Point2D
{
  double x , y;
};



struct Point2DCorrespondance
{
  struct Point2D  * listSource;
  struct Point2D  * listTarget;
  struct Point2D  * depth;
  unsigned int listCurrent;
  unsigned int listMax;
};








static int getPointListNumber(const char * filenameLeft )
{
  fprintf(stderr,"reconstruct3D(%s)\n",filenameLeft);

  ssize_t read;

  FILE * fp = fopen(filenameLeft,"r");
  if (fp!=0)
  {

    char * line = NULL;
    size_t len = 0;
    unsigned int numberOfLines=0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        ++numberOfLines;
    }

    fclose(fp);
    if (line) { free(line); }
    return numberOfLines;
  }

 return 0;
}








static struct Point2DCorrespondance * readPointList(const char * filenameLeft )
{

   struct Point2DCorrespondance * newList=0;


   newList = (struct Point2DCorrespondance *) malloc( sizeof ( struct Point2DCorrespondance )  );

   newList->listMax  =  getPointListNumber(filenameLeft);
   newList->listSource = (struct Point2D *) malloc( sizeof ( struct Point2D ) * newList->listMax  );
   newList->listTarget = (struct Point2D *) malloc( sizeof ( struct Point2D ) * newList->listMax  );
   newList->depth      = (struct Point2D *) malloc( sizeof ( struct Point2D ) * newList->listMax  );
   newList->listCurrent=0;

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

        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", lineStart);


        char * num1 = lineStart; // number1 start to first ' '

        char * num2 = strchr(num1 , ' ');
        while (*num2==' ') { *num2=0; ++num2; }

        char * num3 = strchr(num2 , ' ');
        while (*num3==' ') { *num3=0; ++num3; }

        char * num4 = strchr(num3, ' ');
        while (*num4==' ') { *num4=0; ++num4; }

        //printf("vals are |%s|%s|%s|%s| \n", num1,num2,num3,num4);
        //printf("floats are |%0.2f|%0.2f|%0.2f|%0.2f| \n",atof(num1),atof(num2),atof(num3),atof(num4));

        newList->listSource[newList->listCurrent].x = atof(num1);
        newList->listSource[newList->listCurrent].y = atof(num2);
        newList->listTarget[newList->listCurrent].x = atof(num3);
        newList->listTarget[newList->listCurrent].y = atof(num4);
        ++newList->listCurrent;
    }

    fclose(fp);
    if (line) { free(line); }
    return newList;
}

fprintf(stderr,"Done.. \n");
return 0;
}


#endif // PRIMITIVES_H_INCLUDED
