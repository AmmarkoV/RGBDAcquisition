#ifndef BILATERALFILTER_H_INCLUDED
#define BILATERALFILTER_H_INCLUDED

int bilateralFilter(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                    unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                    float id, float cd , unsigned int dimension
                   );

#endif // BILATERALFILTER_H_INCLUDED
