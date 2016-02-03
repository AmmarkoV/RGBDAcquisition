#ifndef DERICHERECURSIVEGAUSSIAN_H_INCLUDED
#define DERICHERECURSIVEGAUSSIAN_H_INCLUDED


int dericheRecursiveGaussianGrayF(
                                     float * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                     float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                     float *sigma , unsigned int order
                                   );


int dericheRecursiveGaussianGray(
                                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                  float *sigma , unsigned int order
                                );


#endif // DERICHERECURSIVEGAUSSIAN_H_INCLUDED
