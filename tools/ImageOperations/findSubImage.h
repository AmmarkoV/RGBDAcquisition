#ifndef FINDSUBIMAGE_H_INCLUDED
#define FINDSUBIMAGE_H_INCLUDED

int RGBfindImageInImage(
                        char * haystack , unsigned int haystackWidth , unsigned int haystackHeight ,
                        char * needle   , unsigned int needleWidth   , unsigned int needleHeight   ,
                        unsigned int * resX ,
                        unsigned int * resY
                       );

#endif // FINDSUBIMAGE_H_INCLUDED
