#include "findSubImage.h"
#include "patchComparison.h"



int RGBfindImageInImage(
                        unsigned char * haystack , unsigned int haystackWidth , unsigned int haystackHeight ,
                        unsigned char * needle   , unsigned int needleWidth   , unsigned int needleHeight   ,
                        unsigned int * resX ,
                        unsigned int * resY
                       )
{
  unsigned int bestScore=10000000 , bestX=0 , bestY=0;
  unsigned int score=bestScore , x , y;

  for (y=0; y<haystackHeight-needleHeight-1; y++)
  {
    for (x=0; x<haystackWidth-needleWidth-1; x++)
    {
       if (
             compareRGBPatches( haystack , x ,  y , haystackWidth ,  haystackHeight,
                                needle   , 0 ,  0 , needleWidth   ,  needleHeight ,
                                needleWidth   ,  needleHeight
                                , &score)
           )
           {
              if ( score < bestScore) { bestScore = score;  bestX=x;  bestY = y;  }
              if (score < 100 ) { return bestScore+1; }
           }
    }
  }

  *resX = bestX;
  *resY = bestY;
  return bestScore+1;
}
