#include "imageProcessing.h"




int getDepthBlobAverage(float * centerX , float * centerY , float * centerZ , short * frame , unsigned int width , unsigned int height)
{
  unsigned int x=0,y=0;


  unsigned long sumX=0,sumY=0,sumZ=0,samples=0;

   short * sourcePixels   = (short*) frame ;
   short * sourcePixelsEnd   =  sourcePixels + width * height ;

   while (sourcePixels<sourcePixelsEnd)
   {
     if (*sourcePixels != 0)
     {
       sumX+=x;
       sumY+=y;
       sumZ+=*sourcePixels;
       ++samples;
     }
     ++sourcePixels;
     ++x;
     if (x==width) { ++y; x=0;}
   }

   *centerX = (float) sumX / samples;
   *centerY = (float) sumY / samples;
   *centerZ = (float) sumZ / samples;
   return 1;
}





int floodFill(unsigned char * target , unsigned int width , unsigned int height ,
                signed int pX , signed int pY , int threshold,
                unsigned char sR , unsigned char sG , unsigned char sB ,
                unsigned char R , unsigned char G , unsigned char B , int depth)
{
 if ( (pX<0) || (pY<0) || (pX>=width) || (pY>=height) ) { return 0; }
 if (depth>2000) { return 0; }

 if (target==0) { return 0; }
 if (width==0) { return 0; }
 if (height==0) { return 0; }

 unsigned char * source = (unsigned char *) target  + ( (pX*3) + pY * width*3 );

 unsigned char * tR = source; ++source;
 unsigned char * tG = source; ++source;
 unsigned char * tB = source;


  if ( ( *tR == R   ) &&  ( *tG == G   )  &&  ( *tB == B ) ) { return 0; }


  if (
       (( *tR > sR-threshold ) && ( *tR < sR+threshold )) &&
       (( *tG > sG-threshold ) && ( *tG < sG+threshold )) &&
       (( *tB > sB-threshold ) && ( *tB < sB+threshold ))
      )
      {
        *tR = R; *tG = G; *tB = B;

        floodFill(target,width,height, pX+1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX , pY+1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX , pY-1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX+1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX-1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX+1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);
      }

   return 1;
}






int floodFillUShort(unsigned short * target , unsigned int width , unsigned int height ,
                    signed int pX , signed int pY , int threshold,
                    unsigned short sourceDepth ,
                    unsigned short replaceDepth , int depth)
{
 if ( (pX<0) || (pY<0) || (pX>=width) || (pY>=height) ) { return 0; }
 if (depth>2000) { return 0; }

 if (target==0) { return 0; }
 if (width==0) { return 0; }
 if (height==0) { return 0; }

 unsigned short * source = (unsigned short *) target  + ( pX + pY * width );

 unsigned short * tDepth = source; ++source;


  if (  *tDepth == replaceDepth    ) { return 0; }


  if (
       (( *tDepth > sourceDepth-threshold ) && ( *tDepth < sourceDepth+threshold ))
     )
      {
        *tDepth = replaceDepth ;

        floodFillUShort(target,width,height, pX+1 , pY ,   threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX-1 , pY ,   threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX , pY+1 ,   threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX , pY-1 ,   threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX+1 , pY+1 , threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX-1 , pY-1 , threshold, sourceDepth ,replaceDepth ,depth+1);

        floodFillUShort(target,width,height, pX-1 , pY+1 , threshold, sourceDepth ,replaceDepth ,depth+1);
        floodFillUShort(target,width,height, pX+1 , pY-1 , threshold, sourceDepth ,replaceDepth ,depth+1);
      }

   return 1;
}














