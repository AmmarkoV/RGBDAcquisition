#ifndef ACQUISITIONSEGMENT_H_INCLUDED
#define ACQUISITIONSEGMENT_H_INCLUDED


struct SegmentationFeaturesRGB
{
   unsigned int minR ,  minG ,  minB;
   unsigned int maxR , maxG , maxB;

   unsigned int minX , maxX;
   unsigned int minY , maxY;


   unsigned char replaceR , replaceG , replaceB;
   char enableReplacingColors;
};


struct SegmentationFeaturesDepth
{
   unsigned int minDepth, maxDepth;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

};

char * segmentRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf);
short * segmentDepthFrame(short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf);

int getDepthBlobAverage(float * centerX , float * centerY , float * centerZ , short * frame , unsigned int width , unsigned int height);


#endif // ACQUISITIONSEGMENT_H_INCLUDED
