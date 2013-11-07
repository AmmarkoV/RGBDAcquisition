#ifndef FINDSUBIMAGE_H_INCLUDED
#define FINDSUBIMAGE_H_INCLUDED

int RGBfindImageInImage(
                        unsigned char * haystack , unsigned int haystackWidth , unsigned int haystackHeight ,
                        unsigned char * needle   , unsigned int needleWidth   , unsigned int needleHeight   ,
                        unsigned int * resX ,
                        unsigned int * resY
                       );

#endif // FINDSUBIMAGE_H_INCLUDED
