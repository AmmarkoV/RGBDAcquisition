#ifndef DERICHERECURSIVEGAUSSIAN_H_INCLUDED
#define DERICHERECURSIVEGAUSSIAN_H_INCLUDED

int dericheRecursiveGaussianGray(
                                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels,
                                  unsigned char * target,  unsigned int targeteWidth , unsigned int targetHeight ,
                                  float sigma , unsigned int order
                                );

#endif // DERICHERECURSIVEGAUSSIAN_H_INCLUDED
