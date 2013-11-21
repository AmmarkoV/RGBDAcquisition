#ifndef IMAGEPROCESSING_H_INCLUDED
#define IMAGEPROCESSING_H_INCLUDED

void crossProduct(float p1[3] , float p2[3] , float p3[3]  , float * normal);
float dotProduct(float p1[3] , float p2[3] );
float  signedDistanceFromPlane(float origin[3] , float normal[3] , float pN[3]);


int getDepthBlobAverage(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                        unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                        float * centerX , float * centerY , float * centerZ);


int floodFill(unsigned char * target , unsigned int width , unsigned int height ,
                signed int pX , signed int pY , int threshold,
                unsigned char sR , unsigned char sG , unsigned char sB ,
                unsigned char R , unsigned char G , unsigned char B , int depth);


int floodFillUShort(unsigned short * target , unsigned int width , unsigned int height ,
                    signed int pX , signed int pY , int threshold,
                    unsigned short sourceDepth ,
                    unsigned short replaceDepth , int depth);

#endif // IMAGEPROCESSING_H_INCLUDED
