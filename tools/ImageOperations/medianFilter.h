#ifndef MEDIANFILTER_H_INCLUDED
#define MEDIANFILTER_H_INCLUDED


int medianFilter(
                 unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                 unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                 unsigned int blockWidth , unsigned int blockHeight
                );

#endif // MEDIANFILTER_H_INCLUDED
