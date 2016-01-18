#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"

//Stuff from ImageOperations
#include "bilateralFilter.h"
#include "summedAreaTables.h"
#include "imageFilters.h"

int runFilter(int argc, char *argv[])
{
 char * filenameInput=argv[1];
 char * filenameOutput=argv[2];
 unsigned int inputType = guessFilenameTypeStupid(filenameInput);
 struct Image * inputImage = readImage(filenameInput,inputType,0);



 if (inputImage!=0)
 {
    unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
    struct Image * outputImage = copyImage(inputImage);


    if (outputImage!=0)
    {

      unsigned int i=0;

      for (i=0; i<argc; i++)
      {
        if ( strcmp(argv[i],"--bilateral")==0 )
        {
          bilateralFilter( outputImage->pixels ,  outputImage->width , outputImage->height ,
                           inputImage->pixels ,  inputImage->width , inputImage->height ,
                            atof(argv[i+1]) , atof(argv[i+2]) , atoi(argv[i+3])
                         );
        } else
        if ( strcmp(argv[i],"--contrast")==0 )
        {
          contrast(outputImage,atof(argv[i+1]));
        } else
        if ( strcmp(argv[i],"--sattest")==0 )
        {
            //summedAreaTableTest();
            unsigned int * integralImageOutput = 0;
            integralImageOutput = generateSummedAreaTableRGB(inputImage->pixels ,  inputImage->width , inputImage->height);
            if (integralImageOutput!=0)
            {
              free(integralImageOutput);
              fprintf(stderr,"integralImage test was successful\n");
            }
        }
      }








    writeImageFile(outputImage,outputType ,filenameOutput);


    destroyImage(outputImage);
   }
    destroyImage(inputImage);
    return 1;
 }
 return 0;
}



int main(int argc, char *argv[])
{
    if (argc<3)
    {
      fprintf(stderr,"Not enough arguments\n");
      return 1;
    }

    fprintf(stderr,"Converting %s to %s !\n",argv[1],argv[2]);
      runFilter(argc,argv);
    return 0;
}
