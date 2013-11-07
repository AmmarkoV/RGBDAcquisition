#ifndef IMAGEOPS_H_INCLUDED
#define IMAGEOPS_H_INCLUDED

int mixbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height);


int bitbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height);


int mixbltDepth(unsigned short * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
                unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                unsigned int width , unsigned int height);


int saveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);



int saveTileRGBToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                        unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height);

int saveTileDepthToFile(  unsigned int solutionColumn , unsigned int solutionRow ,
                          unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                          unsigned int width , unsigned int height);

int bitBltRGBToFile(  char * name  ,
                      unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                      unsigned int width , unsigned int height);

int bitBltDepthToFile(  char * name  ,
                        unsigned short * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                        unsigned int width , unsigned int height);

#endif // PPM_H_INCLUDED
