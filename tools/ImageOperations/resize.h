#ifndef RESIZE_H_INCLUDED
#define RESIZE_H_INCLUDED


unsigned char * resizeRGBImage(
                               unsigned char * input ,
                               unsigned int originalWidth , unsigned int originalHeight ,
                               unsigned int newWidth , unsigned int newHeight ,
                               unsigned int quality
                               );

#endif // RESIZE_H_INCLUDED

