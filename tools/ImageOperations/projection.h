#ifndef PROJECTION_H_INCLUDED
#define PROJECTION_H_INCLUDED

void createCubeMapFace(
                       char * out , unsigned int outWidth ,unsigned int outHeight , unsigned int outChannels , unsigned int outBitsPerPixel ,
                       char * in , unsigned int inWidth , unsigned int inHeight , unsigned int inChannels , unsigned int inBitsPerPixel
                       );

void getCubeMap2DCoords(unsigned int inputWidth , unsigned int inputHeight , float x ,float y , float z , unsigned int * outX ,unsigned int * outY , unsigned int *outWidth , unsigned int * outHeight );


#endif // PROJECTION_H_INCLUDED
