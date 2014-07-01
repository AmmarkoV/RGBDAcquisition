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

unsigned int learnFaceHistogramHeuristic=0;
int useHistogramHeuristic=0;

unsigned int learnDepthClassifier=0;
int useDepthClassifier=1;

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


 unsigned int minRHistogram[256]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2,1,4,6,4,7,2,5,2,1,8,4,5,7,2,3,2,6,5,6,2,6,3,7,8,5,6,10,8,10,16,19,22,22,14,14,11,30,19,15,22,27,27,28,33,34,26,27,31,37,40,47,46,35,42,43,42,45,46,34,44,42,39,44,44,52,44,47,37,42,38,39,49,45,33,58,42,36,45,45,52,58,66,48,51,56,41,48,36,32,36,36,33,21,28,29,23,22,23,24,18,14,17,9,8,8,15,8,10,10,6,9,9,1,3,1,4,3,0,3,0,1,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 unsigned int minGHistogram[256]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,4,4,5,5,9,9,5,13,10,8,13,14,9,10,13,10,12,17,15,12,15,17,19,27,24,34,29,36,37,37,35,39,37,46,57,47,52,48,47,56,39,54,45,49,71,79,86,85,68,49,55,38,51,43,30,38,31,27,27,27,24,21,28,19,26,18,23,22,19,19,18,21,12,17,15,10,12,10,9,9,11,12,9,3,1,4,3,2,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 unsigned int minBHistogram[256]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,2,0,1,1,0,1,2,1,3,4,4,5,4,6,5,4,4,15,10,15,9,5,8,15,12,16,18,21,21,22,23,19,27,31,33,32,34,36,46,74,50,52,55,45,63,71,30,32,50,41,32,39,48,52,32,29,31,37,26,35,20,23,24,15,13,19,19,16,14,18,12,16,17,10,10,9,8,9,8,9,4,6,6,3,7,4,4,4,4,1,5,1,1,2,1,2,2,0,2,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

 unsigned int maxRHistogram[256]={1087,311,163,54,243,136,48,207,131,112,43,127,175,51,119,133,98,63,141,116,121,21,159,152,47,104,117,99,121,143,171,98,149,135,82,122,147,114,121,113,116,115,90,112,113,84,129,127,126,83,148,98,126,108,77,122,141,113,114,120,123,158,164,80,104,172,156,149,136,170,270,237,187,362,292,213,277,215,236,246,329,267,351,398,237,504,233,348,451,297,363,334,474,375,318,282,394,358,417,252,214,239,279,321,381,264,292,198,196,157,162,193,157,168,252,149,187,208,194,221,267,332,389,306,273,302,299,299,288,284,303,239,207,247,243,206,239,206,216,273,251,202,259,186,190,217,219,206,197,159,216,237,178,179,177,173,149,239,201,214,158,158,163,191,134,185,190,153,134,133,147,107,150,116,136,134,151,193,108,106,98,124,171,135,155,133,118,124,136,144,107,137,109,69,84,77,80,84,73,75,59,60,63,82,77,78,85,83,76,44,66,63,42,48,39,44,46,40,30,41,32,44,31,27,20,25,15,17,21,17,17,12,15,16,13,12,18,14,17,14,13,12,11,13,10,14,18,17,21,22,12,13,30,39,31,471};
 unsigned int maxGHistogram[256]={2896,262,204,210,195,237,267,132,220,252,213,156,251,252,212,205,219,162,186,121,218,186,126,187,128,143,144,184,191,173,242,280,292,213,296,345,297,332,225,359,318,246,296,288,358,303,330,370,373,489,287,478,312,359,311,406,361,372,224,253,282,306,263,251,253,240,209,265,230,232,273,327,374,344,323,323,347,274,299,306,286,324,281,302,264,312,273,265,290,239,297,231,302,255,232,243,210,242,221,211,216,197,203,164,200,189,246,211,154,199,209,179,189,220,154,156,137,171,171,121,160,154,146,140,147,156,156,212,186,214,228,132,194,115,152,122,112,132,134,148,109,119,91,106,99,110,145,109,111,81,90,93,89,68,70,83,58,79,73,67,61,59,68,48,72,75,56,62,49,57,78,70,81,47,36,46,38,41,41,37,43,60,53,55,43,37,48,48,62,65,52,62,35,46,27,28,39,25,18,18,19,16,15,26,16,27,14,14,13,15,14,16,13,13,10,12,13,13,8,9,18,12,15,10,18,15,15,20,16,18,16,26,47,29,29,7,17,9,21,24,34,52,46,32,10,9,5,5,8,8,5,4,7,3,3,471};
 unsigned int maxBHistogram[256]={1312,223,118,479,329,365,239,268,215,176,264,222,330,329,232,229,237,291,271,323,280,232,225,250,236,230,351,394,345,412,332,356,395,337,328,414,456,392,490,237,393,497,456,304,278,421,340,350,336,266,288,290,280,240,209,200,259,275,215,227,216,212,264,279,178,161,264,266,232,269,290,366,284,239,244,260,245,313,229,316,253,329,249,325,272,298,250,319,222,282,254,233,259,272,197,259,214,218,214,242,172,215,179,138,174,168,141,185,161,139,107,98,129,120,130,106,95,92,78,89,113,106,146,100,123,139,148,160,143,230,131,169,152,138,142,101,133,110,128,96,134,85,76,85,73,85,88,79,78,79,77,67,73,68,79,58,68,43,70,55,67,50,41,51,40,36,44,48,39,36,44,45,41,36,34,35,41,30,34,30,35,40,33,47,36,43,39,35,56,43,45,30,52,27,29,19,26,23,22,19,23,27,15,28,15,35,23,16,19,20,15,19,18,13,15,12,15,14,19,30,22,20,12,21,13,17,16,20,25,35,31,16,27,23,20,11,13,9,10,7,7,6,5,9,9,7,9,8,10,4,7,8,8,7,9,471};


 struct depthClassifier dc={0};

void initHistogramLimits()
{
 return;
 unsigned int i=0;
 for (i=0; i<256; i++) {  minRHistogram[i]=10000;   minGHistogram[i]=10000;   minBHistogram[i]=10000; }
}


int storeCalibrationValue(unsigned int tileSize,unsigned int minSize , unsigned int maxSize , unsigned int samples)
{
  headDimensions[tileSize].minSize = minSize;
  headDimensions[tileSize].maxSize = maxSize;
  headDimensions[tileSize].samples = samples;
  return 1;
}

int checkHeadSize(unsigned int tileSize,unsigned int headDepth)
{
   if (!useDepthHeadMinMaxSizeHeuristic) { return 1; }
   if (headDimensions[tileSize].samples<=1) { fprintf(stderr,YELLOW "checkHeadSize(%u,%u) has one/no reference , accepting it..!\n" NORMAL,tileSize,headDepth); return 1; }
   return ( (headDimensions[tileSize].minSize<=headDepth) && (headDepth<=headDimensions[tileSize].maxSize) );
}



void saveClassifierData()
{
 fprintf(stderr,"//-------------- AUTOMATICALLY GENERATED after %u faces --------------\n",faceReadingNumber);
 fprintf(stderr,"void initHeadDimensions()\n");
 fprintf(stderr,"{\n");
 unsigned int i=0;
 for (i=60; i<200; i++)
                { fprintf(stderr,"  storeCalibrationValue(%u,%u,%u,%u);",i,headCountedDimensions[i].minSize,headCountedDimensions[i].maxSize,headCountedDimensions[i].samples);
                  if (i%4==0) { fprintf(stderr,"\n"); }
                }
 fprintf(stderr,"\n}\n");
 fprintf(stderr,"//-------------- --------------\n");

 //-----------------------------------------
 if (learnFaceHistogramHeuristic)
             {
              printOutHistogram((char*) "histogramMinLimit",minRHistogram,minGHistogram,minBHistogram,0);
              printOutHistogram((char*) "histogramMaxLimit",maxRHistogram,maxGHistogram,maxBHistogram,0);
              saveHistogramFilter((char*) "histogramFilter",minRHistogram , minGHistogram , minBHistogram ,
                                                            maxRHistogram , maxGHistogram , maxBHistogram );
             }
  //----------------------------------------
  if (learnDepthClassifier)
             {
               printDepthClassifier((char*) "depthClassifier",&dc);
             }


}



void initHeadDimensions()
{
  storeCalibrationValue(63,1600,3196,4);  storeCalibrationValue(64,1731,3644,13); storeCalibrationValue(65,1575,3240,18); storeCalibrationValue(66,1582,3263,9);
  storeCalibrationValue(67,1483,3179,15); storeCalibrationValue(68,1472,3468,12); storeCalibrationValue(69,1437,2709,6); storeCalibrationValue(70,1372,2568,12);
  storeCalibrationValue(71,1374,2532,12); storeCalibrationValue(72,1354,2038,10); storeCalibrationValue(73,1357,1939,10); storeCalibrationValue(74,1274,2031,7);
  storeCalibrationValue(75,1103,2424,9);  storeCalibrationValue(76,1171,1746,5); storeCalibrationValue(77,1245,1498,9); storeCalibrationValue(78,1241,1770,8);
  storeCalibrationValue(79,1231,1438,6);  storeCalibrationValue(80,1102,1397,8); storeCalibrationValue(81,1105,4664,5); storeCalibrationValue(82,1173,1456,4);
  storeCalibrationValue(83,1170,1295,5);  storeCalibrationValue(84,1157,4748,10); storeCalibrationValue(85,1105,1336,3); storeCalibrationValue(86,1109,5516,6);
  storeCalibrationValue(87,1218,5984,3);  storeCalibrationValue(88,1045,5754,4); storeCalibrationValue(89,1102,1122,4); storeCalibrationValue(90,1062,6019,5);
  storeCalibrationValue(91,1069,5737,7);  storeCalibrationValue(92,996,1223,3); storeCalibrationValue(93,1017,1236,3); storeCalibrationValue(94,973,1178,4);
  storeCalibrationValue(95,985,1102,4);   storeCalibrationValue(96,978,1110,3); storeCalibrationValue(97,971,1119,4); storeCalibrationValue(98,937,1072,4);
  storeCalibrationValue(99,929,1015,4);   storeCalibrationValue(100,900,1044,3); storeCalibrationValue(101,921,1023,4); storeCalibrationValue(102,903,1060,6);
  storeCalibrationValue(103,879,1069,3);  storeCalibrationValue(104,867,960,3); storeCalibrationValue(105,905,955,2); storeCalibrationValue(106,855,995,2);
  storeCalibrationValue(107,806,940,5);   storeCalibrationValue(108,826,887,3); storeCalibrationValue(109,826,916,2); storeCalibrationValue(110,852,916,3);
  storeCalibrationValue(111,798,908,5);   storeCalibrationValue(112,797,885,3); storeCalibrationValue(113,881,970,2); storeCalibrationValue(114,764,888,5);
  storeCalibrationValue(115,801,887,4);   storeCalibrationValue(116,783,912,4); storeCalibrationValue(117,810,910,5); storeCalibrationValue(118,807,812,2);
  storeCalibrationValue(119,777,819,3);   storeCalibrationValue(120,729,799,4); storeCalibrationValue(121,747,844,5); storeCalibrationValue(122,771,856,4);
  storeCalibrationValue(123,760,814,2);   storeCalibrationValue(124,717,863,5); storeCalibrationValue(125,687,803,4); storeCalibrationValue(126,669,842,8);
  storeCalibrationValue(127,661,876,6);   storeCalibrationValue(128,685,863,3); storeCalibrationValue(129,637,764,7); storeCalibrationValue(130,664,756,6);
  storeCalibrationValue(131,658,722,2);   storeCalibrationValue(132,630,739,3); storeCalibrationValue(133,633,728,4); storeCalibrationValue(134,606,689,5);
  storeCalibrationValue(135,608,752,4);   storeCalibrationValue(136,632,720,4); storeCalibrationValue(137,558,651,5); storeCalibrationValue(138,558,695,6);
  storeCalibrationValue(139,525,744,2);   storeCalibrationValue(140,537,678,5); storeCalibrationValue(141,510,684,3); storeCalibrationValue(142,522,662,7);
  storeCalibrationValue(143,552,588,2);   storeCalibrationValue(144,550,612,3); storeCalibrationValue(145,512,585,2); storeCalibrationValue(146,502,527,2);
  storeCalibrationValue(147,502,527,2);   storeCalibrationValue(148,502,519,2);
}


void initDepthClassifier(struct depthClassifier * dc)
{
dc->currentPointList=8;
dc->depthBase=614;
dc->totalSamples=8;
dc->patchWidth=84;
dc->patchHeight=84;

dc->pointList[0].x=42;
dc->pointList[0].y=42;
dc->pointList[0].minAccepted=0;
dc->pointList[0].maxAccepted=0;
dc->pointList[0].samples=101;

dc->pointList[1].x=42;
dc->pointList[1].y=47;
dc->pointList[1].minAccepted=0;
dc->pointList[1].maxAccepted=0;
dc->pointList[1].samples=201;

dc->pointList[2].x=24;
dc->pointList[2].y=32;
dc->pointList[2].minAccepted=27;
dc->pointList[2].maxAccepted=52;
dc->pointList[2].samples=101;

dc->pointList[3].x=60;
dc->pointList[3].y=32;
dc->pointList[3].minAccepted=0;
dc->pointList[3].maxAccepted=53;
dc->pointList[3].samples=101;

dc->pointList[4].x=42;
dc->pointList[4].y=68;
dc->pointList[4].minAccepted=4;
dc->pointList[4].maxAccepted=198;
dc->pointList[4].samples=201;

dc->pointList[5].x=20;
dc->pointList[5].y=54;
dc->pointList[5].minAccepted=17;
dc->pointList[5].maxAccepted=127;
dc->pointList[5].samples=200;

dc->pointList[6].x=64;
dc->pointList[6].y=54;
dc->pointList[6].minAccepted=0;
dc->pointList[6].maxAccepted=45;
dc->pointList[6].samples=201;

dc->pointList[7].x=42;
dc->pointList[7].y=20;
dc->pointList[7].minAccepted=12;
dc->pointList[7].maxAccepted=34;
dc->pointList[7].samples=101;


}


int InitFaceDetection(char * haarCascadePath)
{
     initHistogramLimits();
     initHeadDimensions();
     initDepthClassifier(&dc);

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
 fprintf(stderr,"Head Reading # %u @ frame %u \n", faceReadingNumber , frameNumber);
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

int fitsFaceHistogram(unsigned char * colorPixels ,   unsigned int colorWidth ,unsigned int colorHeight ,
                      unsigned int sX ,unsigned int sY , unsigned int tileWidth ,unsigned int tileHeight )
{
  if (!useHistogramHeuristic) { return 1; }

  unsigned int RHistogram[256]={0};
  unsigned int GHistogram[256]={0};
  unsigned int BHistogram[256]={0};
  unsigned int histogramSamples=0;
  calculateHistogram(colorPixels ,  sX,  sY  , colorWidth,colorHeight,
                     RHistogram ,  GHistogram , BHistogram , &histogramSamples ,
                     tileWidth , tileHeight );

  unsigned int differenceScore=0;
  unsigned int histogramsCompletelyDifferent=0;

 if (learnFaceHistogramHeuristic)
 {
  updateHistogramFilter( RHistogram , GHistogram , BHistogram , &histogramSamples ,
                         minRHistogram , minGHistogram , minBHistogram   ,
                         maxRHistogram , maxGHistogram , maxBHistogram
                       );

 } else
 {
     differenceScore = compareHistogram( RHistogram ,  GHistogram , BHistogram , &histogramSamples ,
                                         minRHistogram ,  minGHistogram ,  minBHistogram ,
                                         maxRHistogram ,  maxGHistogram ,  maxBHistogram  );

    if (differenceScore>4000) { histogramsCompletelyDifferent=1; }
 }

  if (histogramsCompletelyDifferent)
  {
    fprintf(stderr,RED "Discarding bad head #%u due to histogram mismatch ( %u ) \n" NORMAL,discardedFaceByHeuristics,differenceScore);
    if (saveResults)
      {
       fprintf(stderr,YELLOW "Saving Bad Histogram Tile , reminder that you don't want this in production\n" NORMAL );
       char filename[128]={0};
       sprintf(filename,"badHistogram_%u", discardedFaceByHeuristics);
       printOutHistogram(filename,RHistogram,GHistogram,BHistogram,histogramSamples);
      }
     return 0;
  }

  return 1;
}



int fitsFaceDepthClassifier(
                            unsigned short * depthPixels ,   unsigned int depthWidth ,unsigned int depthHeight ,
                            unsigned int sX ,unsigned int sY , unsigned int tileWidth ,unsigned int tileHeight ,
                            unsigned int averageDepth
                           )
{
  if (learnDepthClassifier)
  {
    trainDepthClassifier(&dc,depthPixels,sX,sY,depthWidth,depthHeight,tileWidth,tileHeight);
  } else
  if (useDepthClassifier)
  {
    unsigned int differenceScore = compareDepthClassifier(&dc,depthPixels,sX,sY,depthWidth,depthHeight,tileWidth,tileHeight);

    if  ( differenceScore > 100 )
    {
       fprintf(stderr,RED "Discarding bad head #%u due to bad depth match on depthClassifier ( %u ) \n" NORMAL,discardedFaceByHeuristics,differenceScore);
      return 0;
    }
  }
 return 1;
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
     if ( headCountedDimensions[tileWidth].minSize>averageDepth)
              { headCountedDimensions[tileWidth].minSize=averageDepth; ++headCountedDimensions[tileWidth].samples; }

     if ( headCountedDimensions[tileWidth].maxSize<averageDepth)
              { headCountedDimensions[tileWidth].maxSize=averageDepth; ++headCountedDimensions[tileWidth].samples; }

   }
   fprintf(stderr,"AverageDepth %u has width %u height %u \n",averageDepth,tileWidth,tileHeight);

   if (!checkHeadSize(tileWidth,averageDepth))
   {
     fprintf(stderr,RED "Discarding bad head #%u due to bad tile size heuristic ( tile %u - depth %u ) \n" NORMAL,discardedFaceByHeuristics,tileWidth,averageDepth);
     return 0;
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



         if (
              (fitsFaceDepth(depthPixels ,  depthWidth ,depthHeight , sX , sY , tileWidth , tileHeight , avgFaceDepth )) &&
              (fitsFaceDepthClassifier(depthPixels ,  depthWidth ,depthHeight , sX , sY , tileWidth , tileHeight , avgFaceDepth )) &&
              (fitsFaceHistogram(colorPixels,colorWidth,colorHeight,sX,sY,tileWidth,tileHeight))
            )
         {
          //GOOD Result - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
          //-------------------------------------------------------------------------------------------
          if (saveResults)
          {
            fprintf(stderr,YELLOW "Saving Good Face Tile , reminder that you don't want this in production\n" NORMAL );
            char filename[512]={0};
            sprintf(filename,"faceDetectionR_0_%05u",faceReadingNumber);
            bitBltRGBToFile(filename, 0 , colorPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
            sprintf(filename,"faceDetectionD_0_%05u",faceReadingNumber);
            bitBltDepthToFile(filename, 0 , depthPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
          }
          ++faceReadingNumber;

          if (faceReadingNumber%100==0) {  saveClassifierData(); }

          //Trigger new face detected event
          newFaceDetected(frameNumber,&faceDetected);
          //-------------------------------------------------------------------------------------------
         } else
         {
           //BAD Result - - - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
          //-------------------------------------------------------------------------------------------
          if (saveResults)
          {
            fprintf(stderr,YELLOW "Saving Bad Face Tile , reminder that you don't want this in production\n" NORMAL );
            char filename[512]={0};
            sprintf(filename,"badFaceR_0_%05u",discardedFaceByHeuristics);
            bitBltRGBToFile(filename, 0 , colorPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
            sprintf(filename,"badFaceD_0_%05u",discardedFaceByHeuristics);
            bitBltDepthToFile(filename, 0 , depthPixels , sX, sY  ,640,480 , tileWidth , tileHeight );
          }
            ++discardedFaceByHeuristics;
          //-------------------------------------------------------------------------------------------
         }
    }

    cvReleaseImageHeader( &image );
	return 1;
}

