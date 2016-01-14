#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"

//Stuff from ImageOperations
#include "bilateralFilter.h"
#include "imageFilters.h"

int runFilter(char * filenameInput,char * filenameOutput)
{
 unsigned int inputType = guessFilenameTypeStupid(filenameInput);
 struct Image * inputImage = readImage(filenameInput,inputType,0);



 if (inputImage!=0)
 {
    unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
    struct Image * outputImage = copyImage(inputImage);


    if (outputImage!=0)
    {

      contrast(outputImage,2.4);

      bilateralFilter( outputImage->pixels ,  outputImage->width , outputImage->height ,
                       inputImage->pixels ,  inputImage->width , inputImage->height ,
                       5.0 , 4.0 , 3
                   );


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
    } else
    if (argc>3)
    {
      fprintf(stderr,"Too many arguments\n");
      return 1;
    }

    fprintf(stderr,"Converting %s to %s !\n",argv[1],argv[2]);
      runFilter(argv[1],argv[2]);
    return 0;
}
