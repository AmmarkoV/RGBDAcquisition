#include "compareQuality.h"
#include <math.h>

static inline unsigned char absUCPSub(unsigned char * value1,unsigned char * value2)
{
 if (*value1>*value2) { return (unsigned char) *value1-*value2; }
 return (float ) *value2-*value1;
}


float  calculatePSNR(
                         unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int targetChannels,
                         unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int sourceChannels
                    )
{

   unsigned char * targetPTR = target;
   unsigned char * sourcePTR = source;
   unsigned char * sourceLimit = source + (sourceWidth * sourceHeight * targetChannels);

   if (
        (targetWidth!=sourceWidth)  ||
        (targetHeight!=sourceHeight)  ||
        (targetChannels!=sourceChannels)
       )
   {
     return 10/0.0;
   }


   unsigned int sum = 0;
   while (sourcePTR < sourceLimit)
   {
    unsigned int absd = absUCPSub(sourcePTR,targetPTR);
    sum+=absd * absd;

     ++sourcePTR;
     ++targetPTR;
   }

 float psnr = 10 * log10( (float) (sourceWidth*sourceHeight)  / sum );

 return psnr;
}
