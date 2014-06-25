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

int useDepthHeadMinMaxSizeHeuristic=1;
int saveResults = 1;

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;

unsigned int faceReadingNumber=0;
unsigned int discardedFaceByHeuristics=0;

void * callbackAddr = 0;

struct point2DFace
{
    unsigned int x,y;
};

struct headSizeMinMax
{
    unsigned int minSize,maxSize,samples;
};

struct headSizeMinMax headCountedDimensions[640]={0};
struct headSizeMinMax headDimensions[640]={0};

int storeCalibrationValue(unsigned int tileSize,unsigned int minSize , unsigned int maxSize , unsigned int samples)
{
  headDimensions[tileSize].minSize = minSize;
  headDimensions[tileSize].maxSize = maxSize;
  headDimensions[tileSize].samples = samples;
}

int checkHeadSize(unsigned int tileSize,unsigned int headDepth)
{
   if (!useDepthHeadMinMaxSizeHeuristic) { return 1; }
   if (headDimensions[tileSize].samples<=1) { fprintf(stderr,YELLOW "checkHeadSize(%u,%u) has one/no reference , accepting it..!\n" NORMAL,tileSize,headDepth); return 1; }
   return ( (headDimensions[tileSize].minSize<=headDepth) && (headDepth<=headDimensions[tileSize].maxSize) );
}

void initHeadDimensions()
{
  storeCalibrationValue(63,2500,3196,4);
  storeCalibrationValue(64,1731,3644,13);
  storeCalibrationValue(65,1575,3240,18);
  storeCalibrationValue(66,1582,3263,9);
  storeCalibrationValue(67,1483,3179,15);
  storeCalibrationValue(68,1472,3468,12);
  storeCalibrationValue(69,1437,2709,6);
  storeCalibrationValue(70,1372,2568,12);
  storeCalibrationValue(71,1374,2532,12);
  storeCalibrationValue(72,1354,2038,10);
  storeCalibrationValue(73,1357,1939,10);
  storeCalibrationValue(74,1274,2031,7);
  storeCalibrationValue(75,1103,2424,9);
  storeCalibrationValue(76,1171,1746,5);
  storeCalibrationValue(77,1245,1498,9);
  storeCalibrationValue(78,1241,1770,8);
  storeCalibrationValue(79,1231,1438,6);
  storeCalibrationValue(80,1102,1397,8);
  storeCalibrationValue(81,1105,4664,5);
  storeCalibrationValue(82,1173,1456,4);
  storeCalibrationValue(83,1170,1295,5);
  storeCalibrationValue(84,1157,4748,10);
  storeCalibrationValue(85,1105,1336,3);
  storeCalibrationValue(86,1109,5516,6);
  storeCalibrationValue(87,1218,5984,3);
  storeCalibrationValue(88,1045,5754,4);
  storeCalibrationValue(89,1102,1122,4);
  storeCalibrationValue(90,1062,6019,5);
  storeCalibrationValue(91,1069,5737,7);
  storeCalibrationValue(92,996,1223,3);
  storeCalibrationValue(93,1017,1236,3);
  storeCalibrationValue(94,973,1178,4);
  storeCalibrationValue(95,985,1102,4);
  storeCalibrationValue(96,978,1110,3);
  storeCalibrationValue(97,971,1119,4);
  storeCalibrationValue(98,937,1072,4);
  storeCalibrationValue(99,929,1015,4);
  storeCalibrationValue(100,900,1044,3);
  storeCalibrationValue(101,921,1023,4);
  storeCalibrationValue(102,903,1060,6);
  storeCalibrationValue(103,879,1069,3);
  storeCalibrationValue(104,867,960,3);
  storeCalibrationValue(105,905,955,2);
  storeCalibrationValue(106,855,995,2);
  storeCalibrationValue(107,806,940,5);
  storeCalibrationValue(108,826,887,3);
  storeCalibrationValue(109,826,916,2);
  storeCalibrationValue(110,852,916,3);
  storeCalibrationValue(111,798,908,5);
  storeCalibrationValue(112,797,885,3);
  storeCalibrationValue(113,881,970,2);
  storeCalibrationValue(114,764,888,5);
  storeCalibrationValue(115,801,887,4);
  storeCalibrationValue(116,783,912,4);
  storeCalibrationValue(117,810,910,5);
  storeCalibrationValue(118,807,812,2);
  storeCalibrationValue(119,777,819,3);
  storeCalibrationValue(120,729,799,4);
  storeCalibrationValue(121,747,844,5);
  storeCalibrationValue(122,771,856,4);
  storeCalibrationValue(123,760,814,2);
  storeCalibrationValue(124,717,863,5);
  storeCalibrationValue(125,687,803,4);
  storeCalibrationValue(126,669,842,8);
  storeCalibrationValue(127,661,876,6);
  storeCalibrationValue(128,685,863,3);
  storeCalibrationValue(129,637,764,7);
  storeCalibrationValue(130,664,756,6);
  storeCalibrationValue(131,658,722,2);
  storeCalibrationValue(132,630,739,3);
  storeCalibrationValue(133,633,728,4);
  storeCalibrationValue(134,606,689,5);
  storeCalibrationValue(135,608,752,4);
  storeCalibrationValue(136,632,720,4);
  storeCalibrationValue(137,558,651,5);
  storeCalibrationValue(138,558,695,6);
  storeCalibrationValue(139,525,744,2);
  storeCalibrationValue(140,537,678,5);
  storeCalibrationValue(141,510,684,3);
  storeCalibrationValue(142,522,662,7);
  storeCalibrationValue(143,552,588,2);
  storeCalibrationValue(144,550,612,3);
  storeCalibrationValue(145,512,585,2);
  storeCalibrationValue(146,502,527,2);
  storeCalibrationValue(147,502,527,2);
  storeCalibrationValue(148,502,519,2);
}


unsigned short getDepthValueAtXY(unsigned short * depthFrame ,unsigned int width , unsigned int height ,unsigned int x2d, unsigned int y2d )
{
    if (depthFrame == 0 ) {  return 0; }
    if ( (x2d>=width) || (y2d>=height) )    {   return 0; }


    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    unsigned short result = * depthValue;

    return result;
}



int InitFaceDetection(char * haarCascadePath)
{
     initHeadDimensions();

    /* load the classifier
       note that I put the file in the same directory with
       this code */
    cascade = ( CvHaarClassifierCascade* ) cvLoad(haarCascadePath, 0, 0, 0 );
    if (cascade==0) { fprintf(stderr,"Could not load cascade file %s\n",haarCascadePath); return 0; }
    /* setup memory buffer; needed by the face detector */
    storage = cvCreateMemStorage( 0 );
    if (storage==0) { fprintf(stderr,"Could not allocate storage \n"); return 0; }
    return 1;
}

int CloseFaceDetection()
{
    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

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





int fitsFaceDepth(unsigned short * depthPixels ,   unsigned int depthWidth ,unsigned int depthHeight ,
                  unsigned int sX ,unsigned int sY , unsigned int tileWidth ,unsigned int tileHeight ,
                  unsigned int averageDepth
                  )
{
   if (headCountedDimensions[tileWidth].samples==0)
   {
     headCountedDimensions[tileWidth].minSize=averageDepth;
     headCountedDimensions[tileWidth].maxSize=averageDepth;
     headCountedDimensions[tileWidth].samples=1;
   } else
   {
     if ( headCountedDimensions[tileWidth].minSize>averageDepth) { headCountedDimensions[tileWidth].minSize=averageDepth; ++headCountedDimensions[tileWidth].samples; }
     if ( headCountedDimensions[tileWidth].maxSize<averageDepth) { headCountedDimensions[tileWidth].maxSize=averageDepth; ++headCountedDimensions[tileWidth].samples; }

   }
   fprintf(stderr,"AverageDepth %u has width %u height %u \n",averageDepth,tileWidth,tileHeight);

   if (!checkHeadSize(tileWidth,averageDepth))
   {
     fprintf(stderr,RED "Discarding head due to bad tile size heuristic ( tile %u - depth %u ) \n" NORMAL,tileWidth,averageDepth);
     return 0;
   }

 return 1;

 #define MAX_AREAS_FOR_DEPTH_CHECK 10
 struct point2DFace hotAreas[MAX_AREAS_FOR_DEPTH_CHECK]={0};
 int i=0;
 //84x84
 //Nose Center
 hotAreas[i].x=42; hotAreas[i].y=42; ++i;
 //Nose Down
 hotAreas[i].x=42; hotAreas[i].y=47; ++i;
 //RightEye
 hotAreas[i].x=24; hotAreas[i].y=32; ++i;
 //LefttEye
 hotAreas[i].x=60; hotAreas[i].y=32; ++i;
 //Chin
 hotAreas[i].x=42; hotAreas[i].y=78; ++i;


 //Right Cheek
 hotAreas[i].x=20; hotAreas[i].y=54; ++i;
 //Left Cheek
 hotAreas[i].x=64; hotAreas[i].y=54; ++i;

 //Forehead
 hotAreas[i].x=42; hotAreas[i].y=20; ++i;

 unsigned int answerX=105,answerY=105;
 signed int answers[MAX_AREAS_FOR_DEPTH_CHECK]={854,851,845,843,843,854,843};

 signed int results[MAX_AREAS_FOR_DEPTH_CHECK]={0};

 float x,y;
 unsigned int z=0;
 for (z=0; z<i; z++)
 {
   x = hotAreas[z].x * tileWidth / 84;
   y = hotAreas[z].y * tileHeight /84;

   hotAreas[z].x=(unsigned int ) x;
   hotAreas[z].y=(unsigned int ) y;

   results[z]=getDepthPixel(depthPixels,depthWidth,depthHeight,sX+hotAreas[z].x,sY+hotAreas[z].y);
   fprintf(stderr,"Hot Area %u is %u \n",z,results[z]);

   if (z>0)
   {
      fprintf(stderr,"Number is %i ( %i ) \n",results[z]-results[z-1],answers[z]-answers[z-1]);
   }
   setDepthPixel(depthPixels,depthWidth,depthHeight,sX+hotAreas[z].x,sY+hotAreas[z].y,0);

 }
 return 1;
}












unsigned int DetectFaces(unsigned int frameNumber ,
                         unsigned char * colorPixels ,  unsigned int colorWidth ,unsigned int colorHeight ,
                         unsigned short * depthPixels ,   unsigned int depthWidth ,unsigned int depthHeight ,
                         struct calibration * calib ,
                         unsigned int maxHeadSize,unsigned int minHeadSize)
{
    if  (colorPixels == 0 )  { return 0; }
    if (cascade==0)  { return 0; }
    if (storage==0)  { return 0; }

    /* detect faces */
    IplImage  *image=0;
    image = cvCreateImageHeader( cvSize(colorWidth,colorHeight), IPL_DEPTH_8U, 3 );
    if (image==0) { fprintf(stderr,"Could not allocate image \n"); return 0; }
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
         unsigned int avgFaceDepth = countDepthAverage(depthPixels,depthWidth,depthHeight,sX,sY,tileWidth,tileHeight);


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



         if (fitsFaceDepth(depthPixels ,  depthWidth ,depthHeight , sX , sY , tileWidth , tileHeight , avgFaceDepth ))
         {

          if (saveResults)
          {
            char filename[512]={0};
            sprintf(filename,"faceDetectionR_0_%05u",faceReadingNumber);
            bitBltRGBToFile(filename, 0 , colorPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
            sprintf(filename,"faceDetectionD_0_%05u",faceReadingNumber);
            bitBltDepthToFile(filename, 0 , depthPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
          }
         ++faceReadingNumber;

          if (faceReadingNumber%100==0)
          {
             fprintf(stderr,"//-----------AUTOMATICALLY GENERATED after %u faces --------------\n",faceReadingNumber);
             fprintf(stderr,"void initHeadDimensions()\n");
             fprintf(stderr,"{\n");
             unsigned int i=0;
             for (i=60; i<200; i++)
             {
                 fprintf(stderr,"  storeCalibrationValue(%u,%u,%u,%u);\n",i,headCountedDimensions[i].minSize,headCountedDimensions[i].maxSize,headCountedDimensions[i].samples);
             }
             fprintf(stderr,"}\n");
             fprintf(stderr,"//----------- --------------\n");
          }


          newFaceDetected(frameNumber,&faceDetected);
         } else
         {
          if (saveResults)
          {
            char filename[512]={0};
            sprintf(filename,"badFaceR_0_%05u",discardedFaceByHeuristics);
            bitBltRGBToFile(filename, 0 , colorPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
            sprintf(filename,"badFaceD_0_%05u",discardedFaceByHeuristics);
            bitBltDepthToFile(filename, 0 , depthPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
          }
            ++discardedFaceByHeuristics;
         }
    }


    cvReleaseImageHeader( &image );

	return 1;
}

