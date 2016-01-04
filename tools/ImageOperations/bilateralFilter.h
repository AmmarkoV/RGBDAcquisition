#ifndef BILATERALFILTER_H_INCLUDED
#define BILATERALFILTER_H_INCLUDED


int bilateralFilter(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                    unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                    unsigned char * convolutionMatrix , unsigned int convolutionMatrixWidth , unsigned int convolutionMatrixHeight , unsigned int divisor ,

                    unsigned int tX,  unsigned int tY  ,
                    unsigned int sX,  unsigned int sY  ,
                    unsigned int patchWidth , unsigned int patchHeight
                   );

#endif // BILATERALFILTER_H_INCLUDED
