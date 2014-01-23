#include "FaceDetection.h"
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <cxcore.h>
#include "../tools/ImageOperations/imageOps.h"



#define NORMAL "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */

#define MEMPLACE1(x,y,width) ( y * ( width  ) + x )

IplImage  *image=0;
char * opencv_pointer_retainer=0; // This is a kind of an ugly hack ( see lines noted with UGLY HACK ) to minimize memcpying between my VisCortex and OpenCV , without disturbing OpenCV

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;

unsigned int fdFrameWidth = 640;
unsigned int fdFrameHeight = 480;

void * callbackAddr = 0;


unsigned short getDepthValueAtXY(unsigned short * depthFrame ,unsigned int width , unsigned int height ,unsigned int x2d, unsigned int y2d )
{
    if (depthFrame == 0 ) {  return 0; }
    if ( (x2d>=width) || (y2d>=height) )    {   return 0; }


    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    unsigned short result = * depthValue;

    return result;
}






int getDepthBlobAverage(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                        unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                        float * centerX , float * centerY , float * centerZ)
{

  if (frame==0)  { return 0; }
  if ( (width==0)||(height==0) ) { return 0; }
  if ( (frameWidth==0)||(frameWidth==0) ) { return 0; }

  if (sX>=frameWidth) { return 0; }
  if (sY>=frameHeight) { return 0;  }

  //Check for bounds -----------------------------------------
  if (sX+width>=frameWidth) { width=frameWidth-sX;  }
  if (sY+height>=frameHeight) { height=frameHeight-sY;  }
  //----------------------------------------------------------


  unsigned int x=0,y=0;
  unsigned long sumX=0,sumY=0,sumZ=0,samples=0;

  unsigned short * sourcePTR      = frame+ MEMPLACE1(sX,sY,frameWidth);
  unsigned short * sourceLimitPTR = frame+ MEMPLACE1((sX+width),(sY+height),frameWidth);
  unsigned short sourceLineSkip = (frameWidth-width)  ;
  unsigned short * sourceLineLimitPTR = sourcePTR + (width);

  while (sourcePTR < sourceLimitPTR)
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
       if (*sourcePTR!=0)
       {
        sumX+=x;
        sumY+=y;
        sumZ+=*sourcePTR;
        ++samples;
       }

       ++x;
       ++sourcePTR;
     }

    x=0; ++y;
    sourceLineLimitPTR+=frameWidth;
    sourcePTR+=sourceLineSkip;
  }


   *centerX = (float) sumX / samples;
   *centerY = (float) sumY / samples;
   *centerZ = (float) sumZ / samples;
   return 1;
}




int InitFaceDetection(char * haarCascadePath , unsigned int width ,unsigned int height)
{
    /* load the classifier
       note that I put the file in the same directory with
       this code */
    cascade = ( CvHaarClassifierCascade* ) cvLoad(haarCascadePath, 0, 0, 0 );
    if (cascade==0) { fprintf(stderr,"Could not load cascade file %s\n",haarCascadePath); return 0; }
    /* setup memory buffer; needed by the face detector */
    storage = cvCreateMemStorage( 0 );
    if (storage==0) { fprintf(stderr,"Could not allocate storage \n"); return 0; }

    image = cvCreateImage( cvSize(width,height), IPL_DEPTH_8U, 3 );
    if (image==0) { fprintf(stderr,"Could not allocate image \n"); return 0; }

    fdFrameWidth = width;
    fdFrameHeight = height;


    opencv_pointer_retainer = image->imageData; // UGLY HACK
    return 1;
}

int CloseFaceDetection()
{
    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

    image->imageData = opencv_pointer_retainer; // UGLY HACK
    cvReleaseImage( &image );
    return 1;
}


int registerFaceDetectedEvent(void * callback)
{
  callbackAddr = callback;
  return 1;
}



void newFaceDetected(unsigned int frameNumber ,
                    unsigned int sX , unsigned int sY , unsigned int tileWidth , unsigned int tileHeight ,
                    unsigned int distance ,
                    float headX,float headY,float headZ
                   )
{
 fprintf(stderr, BLUE " " );
 fprintf(stderr,"-----------------------------\n");
 fprintf(stderr,"Head Reading @ frame %u \n", frameNumber);
 fprintf(stderr,"HeadProjection  @ %u %u , %u , %u   \n", sX , sY , tileWidth , tileHeight );
 fprintf(stderr,"Head @ 3D %0.2f %0.2f %0.2f  \n",headX , headY , headZ);
 fprintf(stderr,"Head Distance @  %u\n",distance);
 fprintf(stderr,"-----------------------------\n");
 fprintf(stderr,  " \n" NORMAL );


 if (callbackAddr!=0)
 {
   void ( *DoCallback) (  unsigned int  , unsigned int  , unsigned int , unsigned int , unsigned int , unsigned int , float  ,float  ,float  )=0 ;
   DoCallback = (void(*) (  unsigned int  , unsigned int  , unsigned int , unsigned int , unsigned int , unsigned int , float  ,float  ,float  ) ) callbackAddr;
   DoCallback(frameNumber , sX , sY , tileWidth , tileHeight , distance , headX, headY, headZ);
 }

 return;
}

unsigned int DetectFaces(unsigned int frameNumber , unsigned char * colorPixels , unsigned short * depthPixels, struct calibration * calib ,unsigned int maxHeadSize,unsigned int minHeadSize)
{
    if  (colorPixels == 0 )  { return 0; }
    if (cascade==0)  { return 0; }
    if (storage==0)  { return 0; }
    if (image==0)    { return 0; }

    /* detect faces */
    image->imageData=(char*) colorPixels; // UGLY HACK

    unsigned int maxX=(unsigned int) (minHeadSize)       , maxY=minHeadSize;
    unsigned int minX=(unsigned int) (0.77*maxHeadSize)  , minY=maxHeadSize;
    //fprintf(stderr,"Detect Faces Min %u,%u Max %u,%u \n",minX,minY,maxX,maxY);

    CvSeq *faces = cvHaarDetectObjects
           (
            image,
            cascade,
            storage,
            1.1,
            3,
            0 /*CV_HAAR_DO_CANNY_PRUNNING*/
            , cvSize(minX,minY)
            , cvSize(maxX,maxY) // <--- This might have to be commented out if compiled with C++ :P
            );

    /* for each face found, draw a red box */
    int i;
    for( i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ )
    {
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );

         unsigned int tileWidthHalf = (unsigned int) r->width/2 , tileHeightHalf = (unsigned int) r->height/2;
         unsigned int tileWidth = (unsigned int) r->width , tileHeight = (unsigned int) r->height;
         unsigned int sX = r->x;
         unsigned int sY = r->y;
         unsigned int avgFaceDepth = countDepthAverage(depthPixels,fdFrameWidth,fdFrameHeight,sX,sY,tileWidth,tileHeight);

         /*
         float centerX , centerY , centerZ;
         getDepthBlobAverage(depthPixels,fdFrameWidth,fdFrameHeight,
                             sX,sY,tileWidth,tileHeight,
                             &centerX , &centerY , &centerZ);*/

         float headX=0.0 , headY=0.0 , headZ=0.0;
         transform2DProjectedPointTo3DPoint(calib, sX+tileWidthHalf , sY+tileHeightHalf , (unsigned short) avgFaceDepth , &headX , &headY , &headZ);

         newFaceDetected(frameNumber,sX,sY,tileWidth,tileHeight,avgFaceDepth ,headX,headY,headZ);

    }

	return 1;
}

