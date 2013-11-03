#ifndef PATCHCOMPARISON_H_INCLUDED
#define PATCHCOMPARISON_H_INCLUDED

unsigned int compareDepthPatches( unsigned short * patchADepth , unsigned int pACenterX,  unsigned int pACenterY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                                  unsigned short * patchBDepth , unsigned int pBCenterX,  unsigned int pBCenterY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                                  unsigned int patchWidth, unsigned int patchHeight );

#endif // PATCHCOMPARISON_H_INCLUDED
