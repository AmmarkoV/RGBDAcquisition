#include "codecs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codecs.h"
#include "ppmInput.h"

#define PRINT_COMMENTS 1
#define PPMREADBUFLEN 256


unsigned char * ReadASCIIRaw(unsigned char * buffer , char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp)
{
    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"r");

    if (pf!=0 )
    {
        *width=0; *height=0; *timestamp=0;

        //Todo read once
         //if (pixels==0) {  pixels= (unsigned char*) malloc(w*h*bytesPerPixel*channels*sizeof(char)); }


        fclose(pf);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }
  return buffer;
}


int ReadASCII(char * filename,struct Image * pic,char read_only_header)
{
  pic->pixels = ReadASCIIRaw(pic->pixels , filename, &pic->width, &pic->height, &pic->timestamp );
  return (pic->pixels!=0);
}



int WriteASCII(char * filename,struct Image * pic,int packed)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);
    if (pic==0) { return 0; }
    if ( (pic->width==0) || (pic->height==0) || (pic->channels==0) || (pic->bitsperpixel==0) )
        {
          fprintf(stderr,"WriteASCII(%s) called with zero dimensions ( %ux%u %u channels %u bpp\n",filename,pic->width , pic->height,pic->channels,pic->bitsperpixel);
          return 0;
        }
    if(pic->pixels==0) { fprintf(stderr,"WriteASCII(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"w");

    if (fd!=0)
    {

      char * ptr = pic->pixels ;
      unsigned int x , y;

        if (packed)
        {
           for (y=0; y<pic->height; y++)
           {
            for (x=0; x<pic->width; x++)
             {
               fprintf(fd, "%u ",*ptr);
               ++ptr;
             }
             fprintf(fd, "%\n ");
           }
        } else
        {
         unsigned int channel=0;
         for (channel=0; channel<3; channel++)
         {
           ptr = pic->pixels + channel;
           for (y=0; y<pic->height; y++)
           {
            for (x=0; x<pic->width; x++)
             {
               fprintf(fd, "%u ",*ptr);
               ptr+=3;
             }
             fprintf(fd, "%\n ");
           }
         }
        }

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








