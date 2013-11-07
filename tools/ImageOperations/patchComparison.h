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

#endif // PATCHCOMPARISON_H_INCLUDED
