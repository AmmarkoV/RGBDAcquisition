#ifndef COMPAREQUALITY_H_INCLUDED
#define COMPAREQUALITY_H_INCLUDED


float  calculatePSNR(
                         unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels,
                         unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels
                    );

#endif // COMPAREQUALITY_H_INCLUDED
