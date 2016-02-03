#ifndef CONSTANTTIMEBILATERALFILTERING_H_INCLUDED
#define CONSTANTTIMEBILATERALFILTERING_H_INCLUDED

int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float * sigma ,
                                unsigned int bins ,
                                int useDeriche
                               );

#endif // CONSTANTTIMEBILATERALFILTERING_H_INCLUDED
