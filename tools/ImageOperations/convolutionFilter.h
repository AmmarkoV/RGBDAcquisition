#ifndef CONVOLUTIONFILTER_H_INCLUDED
#define CONVOLUTIONFILTER_H_INCLUDED

int convolutionFilter3ChF(
                       float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                       float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                       float * convolutionMatrix , unsigned int kernelWidth , unsigned int kernelHeight , float divisor
                      );

int convolutionFilter1ChF(
                          float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                          float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          float * convolutionMatrix , unsigned int  kernelWidth , unsigned int kernelHeight , float * divisor
                         );

#endif // CONVOLUTIONFILTER_H_INCLUDED
