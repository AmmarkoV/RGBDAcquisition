#ifndef ACQUISITIONSEGMENT_H_INCLUDED
#define ACQUISITIONSEGMENT_H_INCLUDED


enum combinationModesEnumerator
{
  DONT_COMBINE=0,
  COMBINE_AND   ,
  COMBINE_OR    ,
  COMBINE_XOR   ,
  COMBINE_KEEP_ONLY_RGB  ,
  COMBINE_KEEP_ONLY_DEPTH ,
  //-----------------------------
  NUMBER_OF_COMBINATION_MODES
};


struct floodErasePoints
{
  int totalPoints ;
  int source;
  int target;
  unsigned int pX[32];
  unsigned int pY[32];
  unsigned int threshold[32];
};

struct SegmentationFeaturesRGB
{
   unsigned char eraseColorR , eraseColorG , eraseColorB;

   unsigned int minR ,  minG ,  minB;
   unsigned int maxR , maxG , maxB;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

   unsigned char replaceR , replaceG , replaceB;
   char enableReplacingColors;

   struct floodErasePoints floodErase;
};


struct SegmentationFeaturesDepth
{
   unsigned int minDepth, maxDepth;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

};

unsigned char * selectSegmentationForRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf);
unsigned char * selectSegmentationForDepthFrame(short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf);

int   segmentRGBAndDepthFrame (    char * RGB ,
                                   unsigned short * Depth ,
                                   unsigned int width , unsigned int height ,
                                   struct SegmentationFeaturesRGB * segConfRGB ,
                                   struct SegmentationFeaturesDepth * segConfDepth,
                                   int combinationMode
                               );



#endif // ACQUISITIONSEGMENT_H_INCLUDED
