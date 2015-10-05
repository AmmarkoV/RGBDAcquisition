#include "codecs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codecs.h"
#include "ppmInput.h"

#define PRINT_COMMENTS 1
#define PPMREADBUFLEN 256


unsigned char * ReadASCII(unsigned char * buffer , char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp)
{
    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    if (pixels==0) {  pixels= (unsigned char*) malloc(w*h*bytesPerPixel*channels*sizeof(char)); }
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0; *height=0; *timestamp=0;

        fclose(pf);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }
  return buffer;
}


int ReadASCII(char * filename,struct Image * pic,char read_only_header)
{
  pic->pixels = ReadASCII(pic->pixels , filename, &pic->width, &pic->height, &pic->timestamp );
  return (pic->pixels!=0);
}



int WriteASCII(char * filename,struct Image * pic)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);
    if (pic==0) { return 0; }
    if ( (pic->width==0) || (pic->height==0) || (pic->channels==0) || (pic->bitsperpixel==0) )
        {
          fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions ( %ux%u %u channels %u bpp\n",filename,pic->width , pic->height,pic->channels,pic->bitsperpixel);
          return 0;
        }
    if(pic->pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (pic->bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n"); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (fd!=0)
    {
        unsigned int n;
        if (pic->channels==3) fprintf(fd, "P6\n");
        else if (pic->channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",pic->channels);
            fclose(fd);
            return 1;
        }

        fprintf(fd, "%d %d\n%u\n", pic->width, pic->height , simplePow(2 ,pic->bitsperpixel)-1);

        float tmp_n = (float) pic->bitsperpixel/ 8;
        tmp_n = tmp_n *  pic->width * pic->height * pic->channels ;
        n = (unsigned int) tmp_n;

        fwrite(pic->pixels, 1 , n , fd);
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








