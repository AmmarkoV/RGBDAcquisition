#ifndef NETWORKFRAMEWORK_H_INCLUDED
#define NETWORKFRAMEWORK_H_INCLUDED


int sendImageSocket(int sock , char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel  , unsigned int compressedSize );
char * recvImageSocket(int sock , unsigned int * width , unsigned int * height , unsigned int channels , unsigned int bitsperpixel );

#endif // NETWORKFRAMEWORK_H_INCLUDED
