#ifndef ACQUISITIONSEGMENT_H_INCLUDED
#define ACQUISITIONSEGMENT_H_INCLUDED


struct SegmentationFeaturesRGB
{

   unsigned char minR ,  minG ,  minB;
   unsigned char maxR , maxG , maxB;

   unsigned int minX , maxX;
   unsigned int minY , maxY;


};


#endif // ACQUISITIONSEGMENT_H_INCLUDED
