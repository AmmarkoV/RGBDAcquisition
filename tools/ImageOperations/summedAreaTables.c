#include "summedAreaTables.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int * generateSummedAreaTable(unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight )
{
 if ( (sourceWidth>4000 ) && (sourceHeight>4000) )
 {
   fprintf(stderr,"generateSummedAreaTable cannot contain so big frame sizes \n");
   return 0;
 }


 unsigned int *  sat = (unsigned int * ) malloc(sizeof(unsigned int) * sourceWidth * sourceHeight*3 );
 if (sat==0) { fprintf(stderr,"generateSummedAreaTable could not allocate table\n"); return 0; }

 unsigned int * satOut = sat;


 unsigned char * sourcePtr = source;
 unsigned int nextLine = (sourceWidth*3);
 unsigned char * sourceLineLimit = sourcePtr + nextLine;
 unsigned char * sourceLimit = sourcePtr + (sourceWidth*sourceHeight*3) ;


 //First vertical line is special since it has no vertical additions
 unsigned int lastR=0,lastG=0,lastB=0;
 while (sourcePtr<sourceLimit)
 {
   *satOut = (unsigned int) lastR + (*sourcePtr); lastR=*satOut; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) lastG + (*sourcePtr); lastG=*satOut; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) lastB + (*sourcePtr); lastB=*satOut; ++sourcePtr; ++satOut;
   sourcePtr+=nextLine-3;
 }

 sourcePtr = source;
 satOut = sat;


 //First horizontal line is special since it has no vertical additions
 lastR=0; lastG=0; lastB=0;
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

   *satOut = (unsigned int) *sourcePtr; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) *sourcePtr; ++sourcePtr; ++satOut;
   *satOut = (unsigned int) *sourcePtr; ++sourcePtr; ++satOut;
 }


 return sat;
}
