#include "depthClassifier.h"
#include "imageOps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int trainDepthClassifier(struct depthClassifier * dc ,
                         unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                         unsigned int tileWidth , unsigned int tileHeight)
{
  unsigned int depthValue=0;
  unsigned int i=0;

  unsigned int useX,useY;
  float rateX=1.0,rateY=1.0;
  dc->totalSamples=0;

  for (i=0; i<dc->currentPointList; i++)
  {
     rateX = (float ) dc->pointList[i].x / dc->patchWidth;
     useX = tileWidth * rateX;

     rateY = (float ) dc->pointList[i].y / dc->patchHeight;
     useY = tileHeight * rateY;

     depthValue=getDepthValueAtXY(target,targetWidth,targetHeight ,  tX + useX , tY + useY );

     if (depthValue!=0)
     {
       if (dc->totalSamples==0) { dc->depthBase=depthValue; ++dc->totalSamples; } else
                                             {
                                               if (dc->depthBase > depthValue) { dc->depthBase=depthValue; }
                                               ++dc->totalSamples;
                                             }

      dc->pointList[i].depth=depthValue-dc->depthBase;

      if ( dc->pointList[i].samples==0)
       {
       dc->pointList[i].minAccepted = dc->pointList[i].depth;
       dc->pointList[i].maxAccepted = dc->pointList[i].depth;
        ++dc->pointList[i].samples;
       } else
       {
        if (dc->pointList[i].minAccepted>dc->pointList[i].depth) { dc->pointList[i].minAccepted=dc->pointList[i].depth; }
        if (dc->pointList[i].maxAccepted<dc->pointList[i].depth) { dc->pointList[i].maxAccepted=dc->pointList[i].depth; }
        ++dc->pointList[i].samples;
       }
     }
  }


  for (i=0; i<dc->currentPointList; i++)
  {
      if ( dc->pointList[i].samples!=0)
       {
           dc->pointList[i].depth=dc->pointList[i].depth-dc->depthBase;
       }
  }



 return 1;
}





unsigned int compareDepthClassifier(struct depthClassifier * dc ,
                                    unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                                    unsigned int tileWidth , unsigned int tileHeight)
{
  unsigned int totalDepthMismatch = 0;
  unsigned int i=0;

  dc->totalSamples=0;

  unsigned int useX,useY;
  float rateX=1.0,rateY=1.0;

  for (i=0; i<dc->currentPointList; i++)
  {
     rateX = (float ) dc->pointList[i].x / dc->patchWidth;
     useX = tileWidth * rateX;

     rateY = (float ) dc->pointList[i].y / dc->patchHeight;
     useY = tileHeight * rateY;



     dc->pointList[i].depth = getDepthValueAtXY(target,targetWidth,targetHeight ,  tX + useX , tY + useY );

     setDepthValueAtXY(target,targetWidth,targetHeight,tX + useX,tY + useY,65536);

     if (dc->totalSamples==0) { dc->depthBase = dc->pointList[i].depth; ++dc->totalSamples; } else
                              {
                                if (dc->depthBase >  dc->pointList[i].depth) { dc->depthBase= dc->pointList[i].depth; }
                                ++dc->totalSamples;
                              }
  }


  for (i=0; i<dc->currentPointList; i++)
  {
     if ( (dc->pointList[i].depth!=0) && ( dc->pointList[i].samples>1) )
     {
       dc->pointList[i].depth=dc->pointList[i].depth-dc->depthBase;

       if (dc->pointList[i].maxAccepted<dc->pointList[i].depth) { totalDepthMismatch+=dc->pointList[i].depth - dc->pointList[i].maxAccepted; } else
       if (dc->pointList[i].minAccepted>dc->pointList[i].depth) { totalDepthMismatch+=dc->pointList[i].minAccepted - dc->pointList[i].depth; }
     }
  }
 fprintf(stderr,"Depth Classifier returned %u\n",totalDepthMismatch);
 return totalDepthMismatch;
}



int printDepthClassifier(char * filename , struct depthClassifier * dc )
{
  FILE *fpr = 0;
  fpr=fopen(filename,"w");
  if (fpr!=0)
  {
   fprintf(fpr,"void initDepthClassifier(struct depthClassifier * dc)\n");
   fprintf(fpr,"{ \n");
    unsigned int i=0;
    fprintf(fpr,"dc->currentPointList=%u;\n",dc->currentPointList);
    fprintf(fpr,"dc->depthBase=%u;\n",dc->depthBase);
    fprintf(fpr,"dc->totalSamples=%u;\n",dc->totalSamples);
    fprintf(fpr,"dc->patchWidth=%u;\n",dc->patchWidth);
    fprintf(fpr,"dc->patchHeight=%u;\n",dc->patchHeight);
    for (i=0; i<dc->currentPointList; i++)
     {
       fprintf(fpr,"\ndc->pointList[%u].x=%u;\n",i,dc->pointList[i].x);
       fprintf(fpr,"dc->pointList[%u].y=%u;\n",i,dc->pointList[i].y);
       fprintf(fpr,"dc->pointList[%u].minAccepted=%u;\n",i,dc->pointList[i].minAccepted);
       fprintf(fpr,"dc->pointList[%u].maxAccepted=%u;\n",i,dc->pointList[i].maxAccepted);
       fprintf(fpr,"dc->pointList[%u].samples=%u;\n",i,dc->pointList[i].samples);
     }
    fprintf(fpr,"\n\n");
   fprintf(fpr,"}\n");

   fclose(fpr);
  }
 return 1;
}


