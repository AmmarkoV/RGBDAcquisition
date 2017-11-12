#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"

//Stuff from ImageOperations
#include "bilateralFilter.h"
#include "summedAreaTables.h"
#include "medianFilter.h"
#include "imageFilters.h"
#include "learnImage.h"
#include "special/dericheRecursiveGaussian.h"
#include "tools/imageMatrix.h"
#include "compareQuality.h"

#include "projection.h"


#define TIME_OPERATIONS 1


#if TIME_OPERATIONS
 #include "../Timers/timer.h"
#endif // TIME_OPERATIONS


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

        if ( strcmp(argv[i],"--learn")==0 )
        {
          destroyImage(inputImage);
          learnImage(filenameInput,atoi(argv[i+1]),atoi(argv[i+2]));
          exit(0);
        } else
        if ( strcmp(argv[i],"--rgbcube")==0 )
        {
          unsigned int dim=32;
          unsigned char R = (char) atoi(argv[i]+1);
          unsigned char G = (char) atoi(argv[i]+2);
          unsigned char B = (char) atoi(argv[i]+3);

          outputImage = createImage( dim , dim , 3 , 8 );


           bitbltColorRGB(outputImage->pixels ,   0  ,  0 , dim , dim ,  R ,  G ,  B , dim-1 , dim-1);

           writeImageFile(outputImage,PPM_CODEC ,"new_mX.pnm");
           writeImageFile(outputImage,PPM_CODEC ,"new_pX.pnm");
           writeImageFile(outputImage,PPM_CODEC ,"new_mY.pnm");
           writeImageFile(outputImage,PPM_CODEC ,"new_pY.pnm");
           writeImageFile(outputImage,PPM_CODEC ,"new_mZ.pnm");
           writeImageFile(outputImage,PPM_CODEC ,"new_pZ.pnm");
          destroyImage(outputImage);

        } else
        if ( strcmp(argv[i],"--envcube")==0 )
        {
          fprintf(stdout,"Converting Environment Cube \n");
          unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
          //outputImage = createSameDimensionsImage(inputImage);

          unsigned int outputWidth = inputImage->width;
          unsigned int outputHeight = (unsigned int ) (3*inputImage->width)/4;
          outputImage = createImage( outputWidth , outputHeight , 3 , 8 );

         createCubeMapFace(  outputImage->pixels ,  outputImage->width , outputImage->height , outputImage->channels , outputImage->bitsperpixel ,
                             inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels , inputImage->bitsperpixel
                          );


         struct Image * partImg=0;
         unsigned int outX=0 , outY=0 , outWidth=0 , outHeight=0;
         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/ -1 , /*y*/  0  , /*z*/  0 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_mX.pnm"); destroyImage(partImg);

         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/  1 , /*y*/  0  , /*z*/  0 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_pX.pnm"); destroyImage(partImg);

         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/  0 , /*y*/ -1  , /*z*/  0 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_mY.pnm"); destroyImage(partImg);

         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/  0 , /*y*/  1  , /*z*/  0 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_pY.pnm"); destroyImage(partImg);

         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/  0 , /*y*/  0  , /*z*/ -1 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_mZ.pnm"); destroyImage(partImg);

         getCubeMap2DCoords(outputWidth,outputHeight, /*x*/  0 , /*y*/  0  , /*z*/  1 , &outX , &outY , &outWidth , &outHeight );
         partImg=createImageBitBlt( outputImage , outX , outY , outWidth , outHeight );
         writeImageFile(partImg,PPM_CODEC ,"new_pZ.pnm"); destroyImage(partImg);
        } else
        if ( strcmp(argv[i],"--compare")==0 )
        {
          unsigned int outputType = guessFilenameTypeStupid(filenameOutput);
          outputImage = readImage(filenameOutput,outputType ,0);

          float noise = calculatePSNR( outputImage->pixels ,  outputImage->width , outputImage->height , outputImage->channels ,
                                       inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels );

           fprintf(stdout,"Compared Detected Noise is %0.4f dB \n",noise);
           exit(0);
        } else
        if ( strcmp(argv[i],"--gaussian")==0 )
        {
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);

         unsigned int normalizeGaussianKernel=1;
         unsigned int kernelWidth=5;
         unsigned int kernelHeight=5;
         float * convolutionMatrix=allocateGaussianKernel(kernelWidth,kernelHeight,normalizeGaussianKernel);
         float divisor=1.0;

         float * inF = copyUCharImage2Float(inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels );
         float * outF = (float*) malloc(sizeof(float) *  outputImage->width * outputImage->height *  outputImage->channels );


         convolutionFilter1ChF(
                                 outF ,  outputImage->width , outputImage->height ,
                                 inF,  inputImage->width , inputImage->height ,
                                 convolutionMatrix , kernelWidth , kernelHeight , &divisor
                              );


         free(convolutionMatrix);

         castFloatImage2UChar(outputImage->pixels, outF, outputImage->width , outputImage->height ,  outputImage->channels );
         free(inF);
         free(outF);
        } else
        if ( strcmp(argv[i],"--ctbilateral")==0 )
        {
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);

          float sigma = atof(argv[i+1]);
          constantTimeBilateralFilter(
                                       inputImage->pixels  ,  inputImage->width , inputImage->height , inputImage->channels ,
                                       outputImage->pixels ,  outputImage->width , outputImage->height
                                      ,&sigma //sigma
                                      ,atoi(argv[i+2]) //bins
                                      ,atoi(argv[i+3]) //useDeriche
                                     );

        } else
        if ( strcmp(argv[i],"--deriche")==0 )
        {
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);
          float sigma = atof(argv[i+1]);
          dericheRecursiveGaussianGray( outputImage->pixels ,  outputImage->width , outputImage->height , inputImage->channels ,
                                        inputImage->pixels ,  inputImage->width , inputImage->height ,
                                        &sigma , atoi(argv[i+2])
                                       );
        } else
        if ( strcmp(argv[i],"--dericheF")==0 )
        {
          fprintf(stderr,"This is a test call for casting code , this shouldnt be normally used..\n");
          monochrome(inputImage);
          outputImage = createSameDimensionsImage(inputImage);
          float sigma = atof(argv[i+1]);

          //outputImage = copyImage(inputImage);
         float * inF = copyUCharImage2Float(inputImage->pixels ,  inputImage->width , inputImage->height , inputImage->channels );
         float * outF = (float*) malloc(sizeof(float) *  outputImage->width * outputImage->height *  outputImage->channels );

         dericheRecursiveGaussianGrayF(  outF  ,  outputImage->width , outputImage->height ,  inputImage->channels ,
                                         inF ,  inputImage->width , inputImage->height  ,
                                         &sigma , atoi(argv[i+2])
                                        );

         castFloatImage2UChar(outputImage->pixels, outF, outputImage->width , outputImage->height ,  outputImage->channels );
         free(inF);
         free(outF);
        } else
        if ( strcmp(argv[i],"--median")==0 )
        {
           outputImage = copyImage(inputImage);
           medianFilter3ch(
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
            float * tmp = allocateGaussianKernel(3,5.0,1);
            if (tmp!=0) { free(tmp); }

            tmp = allocateGaussianKernel(9,5.0,1);
            if (tmp!=0) { free(tmp); }


            tmp = allocateGaussianKernel(15,5.0,1);
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


#if TIME_OPERATIONS
 StartTimer(0);
#endif // TIME_OPERATIONS


    fprintf(stdout,"Image Processing %s -> %s !\n",argv[1],argv[2]);
      runFilter(argc,argv);


#if TIME_OPERATIONS
    EndTimer(0);
    fprintf(stdout,"took %u milliseconds !\n",GetLastTimerMilliseconds(0));
#endif // TIME_OPERATIONS

    return 0;
}
