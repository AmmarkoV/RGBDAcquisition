#include "codecs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codecs.h"
#include "ppmInput.h"

#define PRINT_COMMENTS 1
#define PPMREADBUFLEN 256


unsigned char * ReadASCIIRaw(unsigned char * buffer , char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp,int packed)
{
    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *fp=0;
    fp = fopen(filename,"r");

    if (fp!=0 )
    {
        *width=0; *height=0; *timestamp=0;

        fscanf(fp, "%u %u\n",  height , width);
          if (pixels==0)
            {  pixels= (unsigned char*) malloc((*width)*(*height)*8*3*sizeof(char)); }

        if (packed)
        {

        }  else
        {
          unsigned int i,x,y,value=0;
          unsigned char * pixelsPtr = pixels;

          for (i=0; i<3; i++)
          {
           pixelsPtr = pixels+i;
           for (y=0; y<*height; y++)
           {
            for (x=0; x<*width; x++)
             {
               fscanf(fp, "%u ", &value );
               *pixelsPtr=value;
               pixelsPtr+=3;
             }
           }
          }
        }

        fclose(fp);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }
  return pixels;
}


int ReadASCII(char * filename,struct Image * pic,char read_only_header)
{
  pic->pixels = ReadASCIIRaw(pic->pixels , filename, &pic->width, &pic->height, &pic->timestamp , 0 );
  pic->channels=3;
  pic->bitsperpixel=8;
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

      fprintf(fd, "%u %u\n",pic->height,pic->width);

      unsigned char * ptr = pic->pixels ;
      unsigned int x , y;

        if (packed)
        {
           for (y=0; y<pic->height; y++)
           {
            for (x=0; x<pic->width; x++)
             {
               unsigned char value = *ptr;
               fprintf(fd, "%u ",value);
               ++ptr;
             }
             fprintf(fd, "\n ");
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
               unsigned char value = *ptr;
               fprintf(fd, "%u ",value);
               ptr+=3;
             }
             fprintf(fd, "\n ");
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








