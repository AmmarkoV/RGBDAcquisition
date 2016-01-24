#ifndef SUMMEDAREATABLES_H_INCLUDED
#define SUMMEDAREATABLES_H_INCLUDED


unsigned int * generateSummedAreaTableRGB(unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight );



int meanFilterSAT(
                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels ,
                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels ,
                  unsigned int blockWidth , unsigned int blockHeight
                 );

#endif // SUMMEDAREATABLES_H_INCLUDED
