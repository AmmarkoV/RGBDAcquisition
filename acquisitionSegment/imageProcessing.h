#ifndef IMAGEPROCESSING_H_INCLUDED
#define IMAGEPROCESSING_H_INCLUDED



/*
  TAKEN FROM http://www.lighthouse3d.com/opengl/maths/index.php?raytriint

*/

#define innerProduct(v,q) \
       ((v)[0] * (q)[0] + \
		(v)[1] * (q)[1] + \
		(v)[2] * (q)[2])


#define crossProduct(a,b,c) \
        (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
        (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
        (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];



/* a = b - c */
#define vector(a,b,c) \
        (a)[0] = (b)[0] - (c)[0];	\
        (a)[1] = (b)[1] - (c)[1];	\
        (a)[2] = (b)[2] - (c)[2];




#ifdef __cplusplus
extern "C"
{
#endif

float  angleOfNormals(float * p1, float * p2);

void crossProductFrom3Points(float * p1 , float * p2  , float * p3  , float * normal);
float dotProduct(float * p1 , float * p2 );
float  signedDistanceFromPlane(float * origin , float * normal , float * pN );


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


int detectHighContrastUnusableRGB(unsigned char * rgbFrame , unsigned int width , unsigned int height , float percentageHigh);

int detectNoDepth(unsigned short * depthFrame , unsigned int width , unsigned int height , float percentageHigh);


unsigned int countDepths(unsigned short *  depth, unsigned int imageWidth , unsigned int imageHeight  ,
                         unsigned int x , unsigned int y , unsigned int width , unsigned int height ,
                         unsigned int * numberOfHolesIgnored);

int selectVolume(unsigned char * selection ,
                 unsigned short * depthFrame , unsigned int frameWidth , unsigned int frameHeight ,
                 unsigned int sX,unsigned int sY , float sensitivity );


int cutFurtherThanDepth(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                        unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                        unsigned maxDepth );


#ifdef __cplusplus
}
#endif

#endif // IMAGEPROCESSING_H_INCLUDED
