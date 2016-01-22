#include "compareQuality.h"


inline float absFSub(float  value1,float  value2)
{
 if (value1>value2) { return (float ) value1-value2; }
 return (float ) value2-value1;
}


float  calculatePSNR(
                         unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int sourceChannels,
                         unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int targetChannels
                        )
{

   unsigned char * targetPTR = target;
   unsigned char * sourcePTR = source;
   unsigned char * sourceLimit = source + (sourceWidth * sourceHeight * targetChannels);



   while (sourcePTR < sourceLimit)
   {

    //sq_error=abs(I-J).^2;
    //psnr = 10*log10( (h*w)  / sum(sq_error(:)) );


     ++sourcePTR;
     ++targetPTR;
   }

/*
%Input
%A,B -  Images to evaluate / Size must be equal, double type , values[0 255])

[h,w]=size(A);

%Normilize both images
I=A;%/255;
J=B;%/255;


sq_error=abs(I-J).^2;
psnr = 10*log10( (h*w)  / sum(sq_error(:)) );

end*/
}
