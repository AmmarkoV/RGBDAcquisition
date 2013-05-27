#ifndef ACQUISITIONSEGMENT_H_INCLUDED
#define ACQUISITIONSEGMENT_H_INCLUDED


struct SegmentationFeaturesRGB
{
   unsigned char minR ,  minG ,  minB;
   unsigned char maxR , maxG , maxB;

   unsigned int minX , maxX;
   unsigned int minY , maxY;


};


struct SegmentationFeaturesDepth
{
   unsigned int minDepth, maxDepth;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

};

char * segmentRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf);
short * segmentDepthFrame(short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf);


#endif // ACQUISITIONSEGMENT_H_INCLUDED
