#include <stdio.h>
#include <stdlib.h>

#include "../../tools/ImageOperations/imageOps.h"

int main()
{
    printf("Now Testing Image Ops!\n");

    unsigned int width = 640;
    unsigned int height = 480;
    unsigned int rgbSize = width * height * 3;
    unsigned int depthSize = width * height * 1;
    unsigned char * rgb = 0;
    unsigned short * depth = 0;


    unsigned int i=0;

    for (i=0; i<100; i++)
      {
        rgb = ( unsigned char * ) malloc( rgbSize * sizeof(unsigned char));
        depth = ( unsigned char * ) malloc( depthSize * sizeof(unsigned short));

        if ( (rgb!=0) && (depth!=0) )
         {
          memset(rgb,0,rgbSize);
          memset(depth,0,depthSize);

          shiftImageRGB(rgb,rgb,123,123,0, 9 ,  0 , width , height);
          shiftImageDepth(depth,depth, 1 , 9 ,  0  ,  width , height);
         }

        if (rgb!=0)   { free(rgb); }
        if (depth!=0) { free(depth); }
      }



    return 0;
}
