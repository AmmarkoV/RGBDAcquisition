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

 unsigned int *  sat = (unsigned int * ) malloc(sizeof(unsigned int) * sourceWidth * sourceHeight*3 );
 if (sat==0) { fprintf(stderr,"generateSummedAreaTable could not allocate table\n"); return 0; }
 unsigned int * satOut = sat;

 unsigned int memplace=0;
 unsigned int nextLineOffset = (sourceWidth*3);

 unsigned char * inPtr = source;
 unsigned char * inLineLimit = inPtr + nextLineOffset;
 unsigned char * inLimit = inPtr + (sourceWidth*sourceHeight*3) ;

 //First horizontal line is special since it has no vertical additions
 unsigned int pSumR=0,pSumG=0,pSumB=0,y=1;
 while (inPtr<inLineLimit)
 {
   *satOut = (unsigned int) pSumR + (*inPtr); pSumR=*satOut; ++inPtr; ++satOut;
   *satOut = (unsigned int) pSumG + (*inPtr); pSumG=*satOut; ++inPtr; ++satOut;
   *satOut = (unsigned int) pSumB + (*inPtr); pSumB=*satOut; ++inPtr; ++satOut;
   ++memplace;
 }
 inLineLimit+=nextLineOffset;
 fprintf(stderr,"generateSummedAreaTable  done with special first line\n");

 unsigned int *outUpPtr=sat , *outUpLeftPtr=sat , *outLeftPtr=satOut;
 //Ready for the main loop
 while (inPtr<inLimit)
 {
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;
    *satOut = (unsigned int) (*inPtr) + (*outUpPtr); ++inPtr; ++outUpPtr; ++satOut;
    ++memplace;

    while (inPtr<inLineLimit)
    {
      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      fprintf(stderr,"MemR %u = UpLeft = %u , Up = %u , Left = %u , In = %u , Result = %u \n",memplace,(*outUpLeftPtr),(*outUpPtr),(*outLeftPtr),(unsigned int) (*inPtr),(*satOut));
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;

      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      //fprintf(stderr,"MemG %u = UpLeft = %u , Up = %u , Left = %u , In = %u , Result = %u \n",memplace,(*outUpLeftPtr),(*outUpPtr),(*outLeftPtr),(*inPtr),(*satOut));
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;


      *satOut = (unsigned int) (*inPtr) + (*outLeftPtr) +  (*outUpPtr);
      *satOut -= (*outUpLeftPtr);
      //fprintf(stderr,"MemB %u = UpLeft = %u , Up = %u , Left = %u , In = %u , Result = %u \n",memplace,(*outUpLeftPtr),(*outUpPtr),(*outLeftPtr),(*inPtr),(*satOut));
      ++inPtr; ++outUpPtr; ++outUpLeftPtr; ++outLeftPtr; ++satOut;

      ++memplace;
    }
    ++y;
    inLineLimit+=nextLineOffset;
    //outUpLeftPtr+=3;
    outUpLeftPtr=outUpPtr;

   fprintf(stderr,"generateSummedAreaTable  done with %u line\n",y);
 }


 return sat;
}



int summedAreaTableTest()
{
  fprintf(stderr,"summedAreaTableTest()\n");
  unsigned char sourceRGB[]={
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 ,
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 ,
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 ,
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 ,
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 ,
                             1,1,1 , 1,1,1 , 1,1,1 , 1,1,1 , 1,1,1
                          } ;

 unsigned int * output = generateSummedAreaTableRGB(sourceRGB ,  5 , 5 );

 unsigned int x,y,z=0;

 for (y=0; y<5; y++)
 { for (x=0; x<5; x++)
   {
    fprintf(stderr,"%u %u %u ",(unsigned int) sourceRGB[z],(unsigned int) sourceRGB[z+1],(unsigned int) sourceRGB[z+2]);
    z+=3;
   }
   fprintf(stderr,"\n");
 }

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


 return 1;
}
