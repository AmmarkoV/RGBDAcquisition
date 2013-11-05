#include "findSubImage.h"
#include "patchComparison.h"


int RGBfindImageInImage(
                        char * haystack , unsigned int haystackWidth , unsigned int haystackHeight ,
                        char * needle   , unsigned int needleWidth   , unsigned int needleHeight   ,
                        unsigned int * resX ,
                        unsigned int * resY
                       )
{

  unsigned int x,y;

  for (y=200; y<haystackWidth-200; y++)
  {
    for (x=200; x<haystackHeight-200; x++)
    {

       compareDepthPatches( haystack , x ,  y , haystackWidth ,  haystackHeight,
                            needle,  0 ,  0 , needleWidth   ,  needleHeight ,
                                   needleWidth   ,  needleHeight );

    }
  }


  return 0;
}
