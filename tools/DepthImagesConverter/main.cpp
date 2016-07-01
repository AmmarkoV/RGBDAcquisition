/** @file main.cpp
*   @brief  A small tool that converts Depth files from .PNM with swapped endianness to .PNG
*   @author Ammar Qammaz (AmmarkoV)
*   @bug This handles only a few file types
*/

#define USE_JPG_FILES 1
#define USE_PNG_FILES 1
#define USE_OPENCV 0


#if USE_OPENCV
 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui/highgui.hpp>
 using namespace cv;
#endif // USE_OPENCV


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 using namespace std;

#include "../Codecs/codecs.h"


/*
unsigned short * convert24bitTo16bit(void * input24bit,unsigned int width , unsigned int height)
{
  unsigned short * output = (unsigned short * )  malloc ( sizeof(unsigned short)  * width * height  );
  if (output == 0 ) { fprintf(stderr,"Could not allocate output for convert24bitTo16bit\n"); return 0; }

  unsigned char * traverser=(unsigned char * ) input24bit;
  unsigned short * casterUshort=0;
  unsigned int *   casterUint=0;
  unsigned short * outputPointer=output;

  unsigned char * endOfMem = traverser + width * height * 3;
  unsigned short * endOfOutMem = output + width * height;

  unsigned char * byte1;
    unsigned char * byte2;
      unsigned char * byte3;

  while ( ( traverser < endOfMem) && (  outputPointer < endOfOutMem ) )
  {
    byte1 = traverser ++ ;
    byte2 = traverser ++ ;
    byte3 = traverser ++ ;

  // if ( (*byte1!=*byte2) || (*byte2!=*byte3) ) { fprintf(stderr,"!"); }

    casterUshort = (unsigned short*) byte1;
    casterUint = (unsigned int *) byte1;

    //unsigned int outBit = *casterUint;
    // *outputPointer = (unsigned short) outBit;
    *outputPointer = *casterUshort;

    ++outputPointer;
  }

 return output;
}*/


int swapEndianness(struct Image * img)
{
  unsigned char * traverser=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap1=(unsigned char * ) img->pixels;
  unsigned char * traverserSwap2=(unsigned char * ) img->pixels;

  unsigned int bytesperpixel = (img->bitsperpixel/8);
  unsigned char * endOfMem = traverser + img->width * img->height * img->channels * bytesperpixel;

  unsigned char tmp ;
  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    tmp = *traverserSwap1;
    *traverserSwap1 = *traverserSwap2;
    *traverserSwap2 = tmp;

    traverser += bytesperpixel;
  }

 return 1;
}

int readFromPNGDepthAndWriteToPNMDepth(char * inputFilename,char *outputFilename)
{
    struct Image * newImg  = readImage(inputFilename,PNG_CODEC,0);
    if (newImg==0) { fprintf(stderr,"Could not open %s\n",inputFilename); return 0; }

    swapEndianness(newImg);

    fprintf(stderr,"Loaded Image with width : %u , height %u , channels %u , bitsperpixel %u\n",newImg->width,newImg->height,newImg->channels,newImg->bitsperpixel);

    writeImageFile(newImg,PNM_CODEC,outputFilename);

   return 1;
}


int readFromPNGDepthAndWriteToCPNMDepth(char * inputFilename,char *outputFilename)
{
    struct Image * newImg  = readImage(inputFilename,PNG_CODEC,0);
    if (newImg==0) { fprintf(stderr,"Could not open %s\n",inputFilename); return 0; }

    swapEndianness(newImg);

    fprintf(stderr,"Loaded Image with width : %u , height %u , channels %u , bitsperpixel %u\n",newImg->width,newImg->height,newImg->channels,newImg->bitsperpixel);

    writeImageFile(newImg,COMPATIBLE_PNM_CODEC,outputFilename);

   return 1;

}


int readFromPNMDepthAndWriteToPNGDepth(char * inputFilename,char *outputFilename)
{
    struct Image * newImg  = readImage(inputFilename,PNM_CODEC,0);
    if (newImg==0) { fprintf(stderr,"Could not open %s\n",inputFilename); return 0; }
    swapEndianness(newImg);

    fprintf(stderr,"Loaded Image with width : %u , height %u , channels %u , bitsperpixel %u  ... ",newImg->width,newImg->height,newImg->channels,newImg->bitsperpixel);

    if ( writeImageFile(newImg,PNG_CODEC,outputFilename) )
    {
      fprintf(stderr," Success \n");
    } else
    {
      fprintf(stderr," Failed \n");
    }

   return 0;
}




int readFromPNMDepthAndWriteToCPNMDepth(char * inputFilename,char *outputFilename)
{
    struct Image * newImg  = readImage(inputFilename,PNM_CODEC,0);
    if (newImg==0) { fprintf(stderr,"Could not open %s\n",inputFilename); return 0; }

    fprintf(stderr,"Swapping Loaded Image with width : %u , height %u , channels %u , bitsperpixel %u\n",newImg->width,newImg->height,newImg->channels,newImg->bitsperpixel);
    writeImageFile(newImg,COMPATIBLE_PNM_CODEC,outputFilename);

   return 0;
}



int readFromCPNMDepthAndWriteToPNMDepth(char * inputFilename,char *outputFilename)
{
    struct Image * newImg  = readImage(inputFilename,COMPATIBLE_PNM_CODEC,0);
    if (newImg==0) { fprintf(stderr,"Could not open %s\n",inputFilename); return 0; }

    fprintf(stderr,"Swapping Loaded Image with width : %u , height %u , channels %u , bitsperpixel %u\n",newImg->width,newImg->height,newImg->channels,newImg->bitsperpixel);
    writeImageFile(newImg,PNM_CODEC,outputFilename);

   return 0;
}




int main( int argc, char** argv )
{

    fprintf(stderr,"%u arguments\n",argc);
    if( argc < 2)
    {
     cout <<" Usage: DepthImagesConverter inputImage outputImage" << endl;
     return -1;
    }



    if( argc == 3)
    {

     if ( (strstr(argv[1],".png")!=0) && (strstr(argv[2],".pnm")!=0) )
     {
         fprintf(stderr,"Using my custom loader / writer \n");
         return readFromPNGDepthAndWriteToPNMDepth(argv[1],argv[2]);
     } else
     if ( (strstr(argv[1],".png")!=0) && (strstr(argv[2],".cpnm")!=0) )
     {
         fprintf(stderr,"Using my custom loader / writer \n");
         return readFromPNGDepthAndWriteToCPNMDepth(argv[1],argv[2]);
     } else
     if ( (strstr(argv[1],".pnm")!=0) && (strstr(argv[2],".png")!=0) )
     {
       fprintf(stderr,"Using my custom loader / writer from pnm (%s) to png (%s)  \n",argv[1],argv[2]);
       return readFromPNMDepthAndWriteToPNGDepth(argv[1],argv[2]);
     }else
     if ( (strstr(argv[1],".pnm")!=0) && (strstr(argv[2],".cpnm")!=0) )
     {
       fprintf(stderr,"Using my custom loader / writer \n");
       return readFromPNMDepthAndWriteToCPNMDepth(argv[1],argv[2]);
     } else
     if ( (strstr(argv[1],".cpnm")!=0) && (strstr(argv[2],".pnm")!=0) )
     {
       fprintf(stderr,"Using my custom loader / writer \n");
       return readFromCPNMDepthAndWriteToPNMDepth(argv[1],argv[2]);
     } else
     {
       fprintf(stderr,"Could not find an extension combination that can be done with a trick %s -> %s \n",argv[1],argv[2]);
     }

    }


    fprintf(stderr,"Using OpenCV loader / writer \n");



   #if USE_OPENCV
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

//    vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//    compression_params.push_back(9);

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.

    IplImage  *imageDepth = cvCreateImage( cvSize(image.cols,image.rows), IPL_DEPTH_16U , 1 );
    if (imageDepth==0) { fprintf(stderr,"Could not create a new Depth OpenCV Image\n");  return 0; }
    char *opencv_depth_pointer_retainer = imageDepth->imageData; // UGLY HACK
    imageDepth->imageData = (char *) image.data;


    IplImage *rdepth8  = cvCreateImage(cvSize(image.cols,image.rows), IPL_DEPTH_8U, 1);
    cvConvertScaleAbs(imageDepth, rdepth8, 255.0/2048,0);
    cvShowImage("RGBDAcquisition Depth", rdepth8);
    cvReleaseImage( &rdepth8 );

    //cvShowImage("RGBDAcquisition Depth RAW",imageDepth);
    imageDepth->imageData = opencv_depth_pointer_retainer; // UGLY HACK
    cvReleaseImage( &imageDepth );


    imshow( "Display window", image );                   // Show our image inside it.

    if( argc >= 3) { imwrite(argv[2], image  ); } //Convert the image
     else          { waitKey(0); }  // Just show the window
   #else
    fprintf(stderr,"OpenCV Not compiled int , so not doing anything more.. \n");
   #endif // USE_OPENCV


    return 0;
}
