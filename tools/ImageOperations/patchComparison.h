#ifndef PATCHCOMPARISON_H_INCLUDED
#define PATCHCOMPARISON_H_INCLUDED




unsigned int compareDepthPatches( unsigned short * patchADepth , unsigned int pACenterX,  unsigned int pACenterY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                                  unsigned short * patchBDepth , unsigned int pBCenterX,  unsigned int pBCenterY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                                  unsigned int patchWidth, unsigned int patchHeight );


int compareRGBPatches( unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                                unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                                unsigned int patchWidth, unsigned int patchHeight  ,
                                unsigned int * score
                              );

int compareRGBPatchesIgnoreColor
                     ( unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                       unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                       unsigned char ignoreR , unsigned char ignoreG , unsigned char ignoreB ,
                       unsigned int patchWidth, unsigned int patchHeight  ,
                       unsigned int * score,
                       unsigned int  failScore
                     );

int compareRGBPatchesNeighborhoodIgnoreColor
                     (
                       unsigned int neighborhoodX , unsigned int neighborhoodY ,
                       unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                       unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                       unsigned char ignoreR , unsigned char ignoreG , unsigned char ignoreB ,
                       unsigned int patchWidth, unsigned int patchHeight  ,
                       unsigned int * score ,
                       unsigned int  failScore
                     );

unsigned int colorVariance( unsigned char * pixels , unsigned int imageWidth ,unsigned int imageHeight ,
                            unsigned int pX,  unsigned int pY, unsigned int width , unsigned int height);

#endif // PATCHCOMPARISON_H_INCLUDED
