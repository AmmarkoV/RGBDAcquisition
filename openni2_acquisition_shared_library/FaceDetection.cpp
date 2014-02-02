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

int saveResults = 0;

IplImage  *image=0;
char * opencv_pointer_retainer=0; // This is a kind of an ugly hack ( see lines noted with UGLY HACK ) to minimize memcpying between my VisCortex and OpenCV , without disturbing OpenCV

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;

unsigned int fdFrameWidth = 640;
unsigned int fdFrameHeight = 480;
unsigned int faceReadingNumber=0;

void * callbackAddr = 0;


unsigned short getDepthValueAtXY(unsigned short * depthFrame ,unsigned int width , unsigned int height ,unsigned int x2d, unsigned int y2d )
{
    if (depthFrame == 0 ) {  return 0; }
    if ( (x2d>=width) || (y2d>=height) )    {   return 0; }


    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    unsigned short result = * depthValue;

    return result;
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



void newFaceDetected(unsigned int frameNumber , struct detectedFace * faceDetected )
{
 fprintf(stderr, BLUE " " );
 fprintf(stderr,"-----------------------------\n");
 fprintf(stderr,"Head Reading @ frame %u \n", frameNumber);
 fprintf(stderr,"HeadProjection  @ %u %u , %u , %u   \n", faceDetected->sX , faceDetected->sY , faceDetected->tileWidth , faceDetected->tileHeight );
 fprintf(stderr,"Head @ 3D %0.2f %0.2f %0.2f  \n",faceDetected->headX , faceDetected->headY , faceDetected->headZ);
 fprintf(stderr,"Head Distance @  %u\n",faceDetected->distance);
 fprintf(stderr,"-----------------------------\n");
 fprintf(stderr,  " \n" NORMAL );


 if (callbackAddr!=0)
 {
   void ( *DoCallback) (  unsigned int  , struct detectedFace * )=0 ;
   DoCallback = (void(*) (  unsigned int  , struct detectedFace * ) ) callbackAddr;
   DoCallback(frameNumber , faceDetected);
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


         float headX=0.0 , headY=0.0 , headZ=0.0;
         transform2DProjectedPointTo3DPoint(calib, sX+tileWidthHalf , sY+tileHeightHalf , (unsigned short) avgFaceDepth , &headX , &headY , &headZ);


         struct detectedFace faceDetected;
         faceDetected.observationNumber = i;
         faceDetected.observationTotal = faces->total;

         faceDetected.sX = sX;
         faceDetected.sY = sY;
         faceDetected.tileWidth = tileWidth;
         faceDetected.tileHeight = tileHeight;
         faceDetected.distance = avgFaceDepth;
         faceDetected.headX = headX;
         faceDetected.headY = headY;
         faceDetected.headZ = headZ;

         if (saveResults)
          {
            char filename[512]={0};
            sprintf(filename,"colorFrame_0_%05u.pnm",faceReadingNumber);
            bitBltRGBToFile(filename, 0 , colorPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
            ++faceReadingNumber;
          }

         newFaceDetected(frameNumber,&faceDetected);
    }

	return 1;
}

