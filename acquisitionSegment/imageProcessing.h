#ifndef IMAGEPROCESSING_H_INCLUDED
#define IMAGEPROCESSING_H_INCLUDED




int getDepthBlobAverage(float * centerX , float * centerY , float * centerZ , short * frame , unsigned int width , unsigned int height);


int floodFill(unsigned char * target , unsigned int width , unsigned int height ,
                signed int pX , signed int pY , int threshold,
                unsigned char sR , unsigned char sG , unsigned char sB ,
                unsigned char R , unsigned char G , unsigned char B , int depth);

#endif // IMAGEPROCESSING_H_INCLUDED
