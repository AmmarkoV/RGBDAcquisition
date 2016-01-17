#include "summedAreaTables.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int * generateSummedAreaTableRGB(unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight )
{
 fprintf(stderr,"generateSummedAreaTable(%u , %u )\n",sourceWidth , sourceHeight);
 if ( (sourceWidth>4000 ) && (sourceHeight>4000) )
 {
   fprintf(stderr,"generateSummedAreaTable cannot contain so big frame sizes \n");
   return 0;
 }


 unsigned int *  sat = (unsigned int * ) malloc(sizeof(unsigned int) * sourceWidth * sourceHeight*3 );
 if (sat==0) { fprintf(stderr,"generateSummedAreaTable could not allocate table\n"); return 0; }

 unsigned int * satOut = sat;
 unsigned char * sourceUpPtr = source;
 unsigned char * sourcePtr = source;
 unsigned int nextLine = (sourceWidth*3);
 unsigned char * sourceLineLimit = sourcePtr + nextLine;
 unsigned char * sourceLimit = sourcePtr + (sourceWidth*sourceHeight*3) ;

 //First horizontal line is special since it has no vertical additions
 unsigned int lastR=0,lastG=0,lastB=0;
 while (sourcePtr<sourceLineLimit)
 {
   *satOut = (unsigned int) lastR + (*sourcePtr); lastR=*satOut; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) lastG + (*sourcePtr); lastG=*satOut; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) lastB + (*sourcePtr); lastB=*satOut; ++sourcePtr; ++satOut;
 }
 sourceLineLimit+=nextLine;


 //Ready for the main loop
 while (sourcePtr<sourceLimit)
 {
    lastR=0; lastG=0; lastB=0;
    while (sourcePtr<sourceLineLimit)
    {
     /*
       h1 h2 h3 h4 h5 h6 h7 h8
       X1 X2 X3 X4 X5 X6 X7 X8
       X X X X X X X X
       X X X X X X X X
       X X X X X X X X
     */
      *satOut = (unsigned int) lastR + (*sourcePtr) + (*sourceUpPtr); lastR+=(*sourcePtr); ++sourcePtr; ++sourceUpPtr; ++satOut;
      *satOut = (unsigned int) lastG + (*sourcePtr) + (*sourceUpPtr); lastG+=(*sourcePtr); ++sourcePtr; ++sourceUpPtr; ++satOut;
      *satOut = (unsigned int) lastB + (*sourcePtr) + (*sourceUpPtr); lastB+=(*sourcePtr); ++sourcePtr; ++sourceUpPtr; ++satOut;
    }
    sourceLineLimit+=nextLine;
 }


 return sat;
}



int summedAreaTableTest()
{
  fprintf(stderr,"summedAreaTableTest()\n");
  unsigned char sourceRGB={
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
 {
  for (x=0; x<5; x++)
  {
    fprintf(stderr,"%u %u %u ",output[z],output[z+1],output[z+2]);
    z+=3;
  }
 }
 return 1;
}
