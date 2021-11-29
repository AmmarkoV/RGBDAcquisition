#include "save_to_file.h"
#include <stdio.h>
#include <stdlib.h>

#define USE_REGULAR_BYTEORDER_FOR_PNM 1

unsigned int simplePow(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}

int _ogl_swapEndiannessPNM(void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  unsigned char * traverser=(unsigned char * ) pixels;
  unsigned char * traverserSwap1=(unsigned char * ) pixels;
  unsigned char * traverserSwap2=(unsigned char * ) pixels;

  unsigned int bytesperpixel = (bitsperpixel/8);
  unsigned char * endOfMem = traverser + width * height * channels * bytesperpixel;

  unsigned char tmp ;
  while ( ( traverser < endOfMem)  )
  {
    traverserSwap1 = traverser;
    traverserSwap2 = traverser+1;

    tmp = *traverserSwap1;
    *traverserSwap1 = *traverserSwap2;
    *traverserSwap2 = tmp;

    traverser += bytesperpixel;
  }

 return 1;
}

int saveRawImageToFileOGLR(const char * filename,void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperchannel)
{
    if(pixels==0) { fprintf(stderr,"saveRawImageToFileOGLR(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    FILE *fd=0;
    fd = fopen(filename,"wb");

     #if USE_REGULAR_BYTEORDER_FOR_PNM
     //Want Conformance to the NETPBM spec http://en.wikipedia.org/wiki/Netpbm_format#16-bit_extensions
     if (bitsperchannel==16) { _ogl_swapEndiannessPNM(pixels , width , height , channels , bitsperchannel); }
    #else
      #warning "We are using Our Local Byte Order for saving files , this makes things fast but is incompatible with other PNM loaders"
    #endif // USE_REGULAR_BYTEORDER_FOR_PNM

    if (bitsperchannel>16) fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n");
    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            fclose(fd);
            return 1;
        }

        fprintf(fd, "%u %u\n%u\n", width, height , simplePow(2 ,bitsperchannel)-1);

        float tmp_n = (float) bitsperchannel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
        fwrite(pixels, 1 , n , fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }
    return 0;
}
