#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"

//Stuff from ImageOperations
#include "bilateralFilter.h"
#include "summedAreaTables.h"
#include "medianFilter.h"
#include "imageFilters.h"
#include "special/dericheRecursiveGaussian.h"
#include "tools/imageMatrix.h"


int runFilter(int argc, char *argv[])
{
 char * filenameInput=argv[1];
 char * filenameOutput=argv[2];
 unsigned int inputType = guessFilenameTypeStupid(filenameInput);
 struct Image * inputImage = readImage(filenameInput,inputType,0);
 struct Image * outputImage = 0; //This will get allocated when and if needed

 if (inputImage!=0)
 {
    unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
    unsigned int i=0;
      for (i=0; i<argc; i++)
      {



        if ( strcmp(argv[i],"--ctbilateral")==0 )
        {
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);
          //outputImage = copyImage(inputImage);

          constantTimeBilateralFilter(
                                       inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels ,
                                       outputImage->pixels ,  outputImage->width , outputImage->height
                                      ,atof(argv[i+1])  //sigma
                                      ,atoi(argv[i+2]) //bins
                                     );

        } else
        if ( strcmp(argv[i],"--deriche")==0 )
        {
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);
          //outputImage = copyImage(inputImage);

          dericheRecursiveGaussianGray( inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels ,
                                        outputImage->pixels ,  outputImage->width , outputImage->height ,
                                        atof(argv[i+1]) , atoi(argv[i+2])
                                       );
        } else
        if ( strcmp(argv[i],"--dericheF")==0 )
        {
          fprintf(stderr,"This is a test call for casting code , this shouldnt be normally used..\n");
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);

          //outputImage = copyImage(inputImage);
         float * inF = copyUCharImage2Float(inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels );
         float * outF = (float*) malloc(sizeof(float) *  outputImage->width * outputImage->height *  outputImage->channels );

         dericheRecursiveGaussianGrayF( inF ,  inputImage->width , inputImage->height , inputImage->channels ,
                                         outF  ,  outputImage->width , outputImage->height ,
                                         atof(argv[i+1]) , atoi(argv[i+2])
                                        );

         castFloatImage2UChar(outputImage->pixels, outF, outputImage->width , outputImage->height ,  outputImage->channels );
         free(inF);
         free(outF);
        } else
        if ( strcmp(argv[i],"--median")==0 )
        {
           outputImage = copyImage(inputImage);
           medianFilter(
                         outputImage->pixels ,  outputImage->width , outputImage->height ,
                         inputImage->pixels ,  inputImage->width , inputImage->height  ,
                         atoi(argv[i+1]) , atoi(argv[i+2])
                        );
        } else
        if ( strcmp(argv[i],"--meansat")==0 )
        {
           outputImage = copyImage(inputImage);
           meanFilterSAT(
                         outputImage->pixels ,  outputImage->width , outputImage->height , outputImage->channels ,
                         inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels ,
                         atoi(argv[i+1]) , atoi(argv[i+2])
                        );
        } else
        if ( strcmp(argv[i],"--monochrome")==0 )
        {
          outputImage = copyImage(inputImage);
          monochrome(outputImage);
        } else
        if ( strcmp(argv[i],"--bilateral")==0 )
        {
          outputImage = copyImage(inputImage);
          bilateralFilter( outputImage->pixels ,  outputImage->width , outputImage->height ,
                           inputImage->pixels ,  inputImage->width , inputImage->height ,
                            atof(argv[i+1]) , atof(argv[i+2]) , atoi(argv[i+3])
                         );
        } else
        if ( strcmp(argv[i],"--contrast")==0 )
        {
          outputImage = copyImage(inputImage);
          contrast(outputImage,atof(argv[i+1]));
        } else
        if ( strcmp(argv[i],"--sattest")==0 )
        {
            float * tmp = allocateGaussianKernel(3);
            if (tmp!=0) { free(tmp); }

            tmp = allocateGaussianKernel(9);
            if (tmp!=0) { free(tmp); }


            tmp = allocateGaussianKernel(15);
            if (tmp!=0) { free(tmp); }


            summedAreaTableTest();
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

    fprintf(stderr,"Image Processing %s to %s !\n",argv[1],argv[2]);
      runFilter(argc,argv);
    return 0;
}
