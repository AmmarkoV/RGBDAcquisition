#include "FaceDetection.h"
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <cxcore.h>
#include "../tools/Calibration/calibration.h"
#include "../tools/ImageOperations/imageOps.h"


#define MEMPLACE1(x,y,width) ( y * ( width  ) + x )

IplImage  *image=0;
char * opencv_pointer_retainer=0; // This is a kind of an ugly hack ( see lines noted with UGLY HACK ) to minimize memcpying between my VisCortex and OpenCV , without disturbing OpenCV

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;

unsigned int fdFrameWidth = 640;
unsigned int fdFrameHeight = 480;

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



unsigned int DetectFaces(unsigned int frameNumber , unsigned char * colorPixels , unsigned short * depthPixels, unsigned int maxHeadSize,unsigned int minHeadSize)
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


        /*
        if ( settings[REMEMBER_FACES] )
         {
           char timestamped_filename[512]={0};
           timestamped_filename[0]=0; timestamped_filename[1]=0;
           GetANewSnapShotFileName(timestamped_filename,"memfs/faces/face_snap",".ppm");
           SaveRegisterPartToFile(timestamped_filename,vid_reg, r->x , r->y , r->width , r->height );
         }
        */

        fprintf(stderr,"Frame %u : Face%u @ %u , %u ( size %u,%u ) \n",frameNumber,i,r->x , r->y ,  r->width , r->height );
       /* AddToFeatureList(  video_register[vid_reg].faces  ,
                           r->x , r->y , 0 ,
                           r->width , r->height , 0
                         );*/

         //saveFDImageToFile("testD.pnm",(unsigned char*) depthPixels,fdFrameWidth,fdFrameHeight,1,16);
         unsigned int tileWidth = (unsigned int) r->width , tileHeight = (unsigned int) r->height;
         unsigned int sX = r->x;
         unsigned int sY = r->y;
         unsigned int avgDepth = countDepthAverage(depthPixels,fdFrameWidth,fdFrameHeight,sX,sY,tileWidth,tileHeight);
         fprintf(stderr,"AvgDepth %u , Spot Depth %u \n",avgDepth ,(unsigned int)  getDepthValueAtXY(depthPixels,fdFrameWidth,fdFrameHeight,sX,sY));

         float centerX , centerY , centerZ;
         getDepthBlobAverage(depthPixels,fdFrameWidth,fdFrameHeight,
                             sX,sY,tileWidth,tileHeight,
                             &centerX , &centerY , &centerZ);
         fprintf(stderr,"Depth Blob @ %f %f %f  \n",centerX , centerY , centerZ);
         float mouseX , mouseY , mouseZ;
         struct calibration calib;
         NullCalibration(fdFrameWidth,fdFrameHeight,&calib);
         transform2DProjectedPointTo3DPoint(&calib, sX , sY , (unsigned short) centerZ , &mouseX , &mouseY , &mouseZ);
         fprintf(stderr,"Depth Blob Transformed @ %f %f %f  \n", mouseX , mouseY , mouseZ);

    }

	return 1;
}

