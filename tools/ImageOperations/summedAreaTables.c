#include "summedAreaTables.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int * generateSummedAreaTableRGB(unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight )
{
 fprintf(stderr,"generateSummedAreaTable(%p , %u , %u )\n",source,sourceWidth , sourceHeight);
 if ( (sourceWidth>4000 ) && (sourceHeight>4000) )
 {
   fprintf(stderr,"generateSummedAreaTable cannot contain so big frame sizes \n");
   return 0;
 }

 unsigned int *  sat = (unsigned int * ) malloc(sizeof(unsigned int) * ((sourceWidth) * (sourceHeight) *3) );
 if (sat==0) { fprintf(stderr,"generateSummedAreaTable could not allocate table\n"); return 0; }

 unsigned int nextLineOffset = (sourceWidth*3);

 unsigned int * satOut = sat;
 unsigned char * inPtr = source;
 unsigned char * inLineLimit = inPtr + nextLineOffset;
 unsigned char * inLimit = inPtr + (sourceWidth*sourceHeight*3) ;

 //First pixel is just the source ( inPtr ) value , and we go forward
 unsigned int *outLeftPtr=satOut;
 *satOut=(unsigned int) (*inPtr); ++inPtr; ++satOut;
 *satOut=(unsigned int) (*inPtr); ++inPtr; ++satOut;
 *satOut=(unsigned int) (*inPtr); ++inPtr; ++satOut;

 //First horizontal line is special since it has no vertical additions , so we just sum up left and current elements
 while (inPtr<inLineLimit)
 {
   *satOut = (unsigned int) (*inPtr) + (*outLeftPtr); ++inPtr; ++satOut; ++outLeftPtr;
   *satOut = (unsigned int) (*inPtr) + (*outLeftPtr); ++inPtr; ++satOut; ++outLeftPtr;
   *satOut = (unsigned int) (*inPtr) + (*outLeftPtr); ++inPtr; ++satOut; ++outLeftPtr;
 }
 inLineLimit+=nextLineOffset;

 unsigned int *outUpPtr=sat , *outUpLeftPtr=sat;
 //Ready for the main loop
 outLeftPtr=satOut;
 while (inPtr<inLimit)
 {
    outLeftPtr=satOut;
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;

    if (inLineLimit>inLimit) { fprintf(stderr,"Border case\n");  inLineLimit=inLimit;  }
    while (inPtr<inLineLimit)
    {
      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;

      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;


      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;
    }
    inLineLimit+=nextLineOffset;
    outUpLeftPtr=outUpPtr;
 }


 return sat;
}





unsigned int getSATSumRGB(unsigned int *outR , unsigned int *outG , unsigned int *outB  ,
                          unsigned int * sourceSAT ,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          unsigned int x, unsigned int y , unsigned int blockWidth , unsigned int blockHeight )
{
  unsigned int * sourceSATLimit = sourceSAT + (sourceHeight * sourceWidth*3);


  /*                            TOP
       -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
      L-  -  -  -  -  -  A  -  -  -  -  B  -  -  -  -  -  -R
      E-  -  -  -  -  -  -  - REQUEST-  -  -  -  -  -  -  -I
      F-  -  -  -  -  -  -  -   SUM  -  -  -  -  -  -  -  -G
      T-  -  -  -  -  -  C  -  -  -  -  D  -  -  -  -  -  -H
       -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -T
                               BOTTOM
      _____________
      D + A - B - C
  */
  unsigned int * A = sourceSAT + ( sourceWidth*3* (y+0) )           + ((x+0)*3);
  unsigned int * B = sourceSAT + ( sourceWidth*3* (y+0) )           + ((x+blockWidth)*3);
  unsigned int * C = sourceSAT + ( sourceWidth*3* (y+blockHeight) ) + ((x+0)*3);
  unsigned int * D = sourceSAT + ( sourceWidth*3* (y+blockHeight) ) + ((x+blockWidth)*3);

  if (
       (A>=sourceSATLimit) ||
       (B>=sourceSATLimit) ||
       (C>=sourceSATLimit) ||
       (D>=sourceSATLimit)
     ) { return 0; }


  *outR = *D + *A ;
  *outR = *outR - *B - *C;
  ++A; ++B; ++C; ++D; //Next Channel

  *outG = *D + *A ;
  *outG = *outG - *B - *C;
  ++A; ++B; ++C; ++D; //Next Channel

  *outB = *D + *A ;
  *outB = *outB - *B - *C;

  return 1;
}





int meanFilterSAT(
                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels ,
                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels ,
                  unsigned int blockWidth , unsigned int blockHeight
                 )
{
  fprintf(stderr,"meanFilterSAT(%p , %u , %u )\n",source,sourceWidth , sourceHeight);
  if ( (sourceChannels!=3) || (targetChannels!=3) ) {return 0; }

  if (
       (target==0)||
       (targetWidth!=sourceWidth)||
       (targetHeight!=sourceHeight)
     )
{ return 0; }

  unsigned int * sat = generateSummedAreaTableRGB(source,sourceWidth,sourceHeight);
  if (sat==0) { free(target); return 0; }


  fprintf(stderr," doing mean .. \n");
  unsigned int halfBlockWidth = (unsigned int) blockWidth/2;
  unsigned int halfBlockHeight = (unsigned int) blockHeight/2;
  unsigned char * targetPTR = target ;
  unsigned char * targetLimit = target;
  unsigned int  sumR , sumG ,sumB  , blockSize = blockWidth*blockHeight;

  unsigned int leftX=0,leftY=0;
  float res;

  for (leftY=halfBlockHeight; leftY<sourceHeight-halfBlockHeight; leftY++)
  {
    leftX=0;
    targetPTR = target + (leftY*sourceWidth*3) + halfBlockWidth*3 ;
    targetLimit = targetPTR + (sourceWidth*3) -(blockWidth*3) ;
    while (targetPTR<targetLimit)
    {
      getSATSumRGB(&sumR,&sumG,&sumB,sat,sourceWidth,sourceHeight,leftX,leftY,blockWidth,blockHeight);

      res = (float) sumR/blockSize;
      *targetPTR = (unsigned char) res; ++targetPTR;

      res = (float) sumG/blockSize;
      *targetPTR = (unsigned char) res; ++targetPTR;

      res = (float) sumB/blockSize;
      *targetPTR = (unsigned char) res; ++targetPTR;

      ++leftX;
    }
  }

  return 1;
}


int summedAreaTableTest()
{
  fprintf(stderr,"summedAreaTableTest()\n");
  unsigned char sourceRGB[]={
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 ,
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 ,
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 ,
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 ,
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 ,
                             1,0,2 , 1,0,2 , 1,0,2 , 1,0,2 , 1,0,2
                          } ;

 unsigned int * output = generateSummedAreaTableRGB(sourceRGB ,  5 , 5 );
 if (output==0)
 {
   fprintf(stderr,"generateSummedAreaTableRGB failed\n");
   return 0;
 }
 unsigned int x,y,z=0;

 fprintf(stderr,"Input\n");
 for (y=0; y<5; y++)
 { for (x=0; x<5; x++)
   {
    fprintf(stderr,"%u %u %u ",(unsigned int) sourceRGB[z],(unsigned int) sourceRGB[z+1],(unsigned int) sourceRGB[z+2]);
    z+=3;
   }
   fprintf(stderr,"\n");
 }
 fprintf(stderr,"\n\n\n\n\n\n");



 fprintf(stderr,"Output\n");
 z=0;
 for (y=0; y<5; y++)
 { for (x=0; x<5; x++)
   {
    fprintf(stderr,"%u %u %u ",output[z],output[z+1],output[z+2]);
    z+=3;
   }
   fprintf(stderr,"\n");
 }
 fprintf(stderr,"\n\n\n\n\n\n");


  unsigned int outR , outG , outB;
  getSATSumRGB(&outR,&outG,&outB,output,5,5,0,0,5,5);
  fprintf(stderr,"Sum is %u %u %u \n",outR , outG , outB);








 free(output);

 return 1;
}
