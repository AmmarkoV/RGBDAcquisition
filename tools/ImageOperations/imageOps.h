#ifndef IMAGEOPS_H_INCLUDED
#define IMAGEOPS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

void RGBtoHSV( unsigned char r, unsigned char g, unsigned char b,
               float *h, float *s, float *v );

int mixbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height);


int bitbltRGBDebugMode(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
                       unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                       unsigned int width , unsigned int height);

int bitbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height);



int bitbltDepth(unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                unsigned short * source , unsigned int sX,  unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                unsigned int width , unsigned int height);

int mixbltDepth(unsigned short * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
                unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                unsigned int width , unsigned int height);


int printOutHistogram(char * filename, unsigned int * RHistogram_1 , unsigned int * GHistogram_1 , unsigned int * BHistogram_1 , unsigned int Samples_1  );

int updateHistogramFilter(
                           unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                           unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram   ,
                           unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram
                         );


int saveHistogramFilter(
                           char * filename ,
                           unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram   ,
                           unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram
                         );

unsigned int compareHistogram(unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                     unsigned int * minRHistogram , unsigned int * minGHistogram , unsigned int * minBHistogram ,
                     unsigned int * maxRHistogram , unsigned int * maxGHistogram , unsigned int * maxBHistogram  );


int calculateHistogram(unsigned char * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                       unsigned int * RHistogram , unsigned int * GHistogram , unsigned int * BHistogram , unsigned int * samples ,
                       unsigned int width , unsigned int height);


int saveRawImageToFile(char * filename,char * comment ,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);



int saveTileRGBToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                        unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height);

int saveTileDepthToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                          unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                          unsigned int width , unsigned int height);


int shiftImageRGB(unsigned char * target, unsigned char * source ,  unsigned char transR, unsigned char transG, unsigned char transB , signed int tX,  signed int tY  ,  unsigned int width , unsigned int height);
int shiftImageDepth(unsigned short * target, unsigned short * source , unsigned short depthVal  , signed int tX,  signed int tY  ,  unsigned int width , unsigned int height);


int bitbltColorRGB(unsigned char * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                   unsigned char R , unsigned char G , unsigned char B ,
                   unsigned int width , unsigned int height);

int bitbltDepthValue(unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                     unsigned short DepthVal ,
                     unsigned int width , unsigned int height);

int bitBltRGBToFile(  char * name  ,char * comment ,
                      unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                      unsigned int width , unsigned int height);

int bitBltDepthToFile(  char * name  ,char * comment ,
                        unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height);

unsigned int countOccurancesOfRGBPixel(unsigned char * ptrRGB , unsigned int RGBwidth , unsigned int RGBheight , unsigned char transR ,unsigned char transG , unsigned char transB);

int getRGBPixel(unsigned char * ptrRGB  , unsigned int RGBwidth , unsigned int RGBheight ,  unsigned int x , unsigned int y , unsigned char * R , unsigned char * G , unsigned char * B);
unsigned short getDepthPixel(unsigned short * ptrDepth , unsigned int Depthwidth , unsigned int Depthheight ,  unsigned int x , unsigned int y);
int setDepthPixel(unsigned short * ptrDepth , unsigned int Depthwidth , unsigned int Depthheight ,  unsigned int x , unsigned int y , unsigned short depthValue);

int closeToRGB(unsigned char R , unsigned char G , unsigned char B  ,  unsigned char targetR , unsigned char targetG , unsigned char targetB , unsigned int threshold);


unsigned int countDepthAverage(unsigned short * source, unsigned int sourceWidth , unsigned int sourceHeight ,
                                unsigned int sX,  unsigned int sY  , unsigned int tileWidth , unsigned int tileHeight);

#ifdef __cplusplus
}
#endif


#endif // PPM_H_INCLUDED
