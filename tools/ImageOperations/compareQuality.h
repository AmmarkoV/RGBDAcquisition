#ifndef COMPAREQUALITY_H_INCLUDED
#define COMPAREQUALITY_H_INCLUDED


float  calculatePSNR(
                         unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int sourceChannels,
                         unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int targetChannels
                        );

#endif // COMPAREQUALITY_H_INCLUDED
