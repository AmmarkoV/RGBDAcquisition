#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BlobDetector.h"
#include "../../tools/ImageOperations/imageOps.h"

unsigned int framesProcessed=0;



#define MAX_RECURSION_LEVEL 1250



int floodEraseAndGetAverageDepth(unsigned short * depth , unsigned int width , unsigned int height ,
                                 unsigned int x , unsigned int y ,
                                 unsigned int * xSum , unsigned int * ySum ,
                                 unsigned long * depthSum , unsigned int * depthSamples ,
                                 unsigned int recursionLevel , unsigned int * recursionLevelHits
                                 )
{
  if (recursionLevel>=MAX_RECURSION_LEVEL)
  {
    *recursionLevelHits+=1;
    return 0;
  }


  unsigned short * depthVal = depth + x + (width*y) ;
  unsigned short * depthWOff;
  if (*depthVal!=0)
  {
       *depthSum += *depthVal;
       *depthSamples += 1;
       *xSum+=x;
       *ySum+=y;

       *depthVal=0;
  } else
  {
    return 0;
  }



  if (x>0)
  {
   depthWOff=depthVal-1;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x-1,y,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }
  if (y>0)
  {
   depthWOff=depthVal-width;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x,y-1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }




  if (x<width-1)
  {
   depthWOff=depthVal+1;
   if (*depthWOff!=0)
   {
    floodEraseAndGetAverageDepth(depth,width,height,
                                x+1,y,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }
  if (y<height-1)
  {
   depthWOff=depthVal+width;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x,y+1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }





  if ( (x>0)&&(y>0))
  {
   depthWOff=depthVal-width-1;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x-1,y-1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }


  if ( (x>0)&&(y<height-1))
  {
   depthWOff=depthVal+width-1;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x-1,y+1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }


  if ( (x<width-1)&&(y<height-1))
  {
   depthWOff=depthVal+width+1;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x+1,y+1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }


  if ( (x<width-1)&&(y>0))
  {
   depthWOff=depthVal-width+1;
   if (*depthWOff!=0)
   {
   floodEraseAndGetAverageDepth(depth,width,height,
                                x+1,y-1,
                                xSum,ySum,
                                depthSum,depthSamples,
                                recursionLevel+1, recursionLevelHits);
   }
  }



 return 1;
}





struct xyList * extractBlobsFromDepthMap(unsigned short * depth , unsigned int width , unsigned int height , unsigned int maxBlobs , unsigned int minBlobSize)
{
  struct xyList * output = (struct xyList*) malloc(sizeof(struct xyList));
  output->maxListLength=maxBlobs;
  output->listLength=0;

  output->data = (struct xyP*) malloc(maxBlobs * sizeof(struct xyP));

  unsigned short * depthPTR = depth;
  unsigned short * depthLimit = depth + (width*height);
  unsigned int lineOffset = width;
  unsigned short * depthLineLimit = depth + lineOffset;

  unsigned int x=0;
  unsigned int y=0;
  unsigned int avgX=0;
  unsigned int avgY=0;

  unsigned long depthSum;
  unsigned int depthSamples;
  unsigned int recursionLevel;

  unsigned int blobNumber=0;
  unsigned int recursionLevelHits = 0;

  while (depthPTR<depthLimit)
  {
    while (depthPTR<depthLineLimit)
     {
      if (*depthPTR==0)
       {
       } else
       {
         depthSum=0;
         depthSamples=0;
         recursionLevel=0;
         avgX=0;
         avgY=0;


         floodEraseAndGetAverageDepth(depth , width , height , x , y  , &avgX , &avgY, &depthSum , &depthSamples, recursionLevel , &recursionLevelHits);
         if (recursionLevelHits>0)
         {
           fprintf(stderr,"Hit Recursion limits %u times \n",recursionLevelHits);
         }

         if (minBlobSize<=depthSamples)
         {
          float depthValue = (float) depthSum/depthSamples;
          float avgXf = (float) avgX/depthSamples;
          float avgYf = (float) avgY/depthSamples;
          //fprintf(stderr,"Frame %u Found blob #%u  @ %ux%u    ==>  %0.2f,%0.2f  = %0.2f\n" , framesProcessed , blobNumber ,x,y , avgXf, avgYf , depthValue);
          fprintf(stdout,"%u,%u,%0.2f,%0.2f,%0.2f,%u\n" , framesProcessed , blobNumber, avgXf, avgYf , depthValue,depthSamples);

          output->data[blobNumber].x = avgXf;
          output->data[blobNumber].y = avgYf;
          output->data[blobNumber].z = depthValue;

          ++blobNumber;
          output->listLength=blobNumber;
          if (maxBlobs<=blobNumber)
          {
           fprintf(stderr,"Cannot accomodate more than %u blobs\n",maxBlobs);
           return output;
          }
         } else
         {
          fprintf(stderr,"Filtered out blob #%u @ Frame %u  with only %u samples ( min %u )\n" , blobNumber  , framesProcessed , depthSamples ,minBlobSize );

         }
       }
      ++x;
      ++depthPTR;
     }
    depthLineLimit+=lineOffset;
    ++y;
    x=0;
  }

 return output;
}



struct xyList * extractBlobsFromDepthMapNewBuffer(unsigned short * depth , unsigned int width , unsigned int height , unsigned int maxBlobs , unsigned int minBlobSize)
{
  unsigned short * ourcopy = malloc(sizeof(unsigned short) * width * height );
  memcpy(ourcopy,(unsigned short *) depth,sizeof(unsigned short) * width * height  );


   struct xyList *  retres = extractBlobsFromDepthMap( (unsigned short *) ourcopy,width,height,maxBlobs,minBlobSize);

  free(ourcopy);

 return retres;
}






int initArgs_BlobDetector(int argc, char *argv[])
{
 fprintf(stdout,"Frame,Blob,X,Y,Z,Samples\n");
 return 0;

}

int setConfigStr_BlobDetector(char * label,char * value)
{
 return 0;

}

int setConfigInt_BlobDetector(char * label,int value)
{
 return 0;

}



unsigned char * getDataOutput_BlobDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}

int addDataInput_BlobDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{

 fprintf(stderr,"addDataInput_BlobDetector %u (%ux%u)\n" , stream , width, height);
 if (stream==1)
 {
   extractBlobsFromDepthMapNewBuffer( (unsigned short *) data,width,height,128,40);
   ++framesProcessed;

  return 1;
 }
 return 0;
}



unsigned short * getDepth_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}


unsigned char * getColor_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}



int processData_BlobDetector()
{
 return 0;

}


int cleanup_BlobDetector()
{

 return 0;
}


int stop_BlobDetector()
{
 return 0;

}

