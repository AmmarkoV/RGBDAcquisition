#include "codecs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "codecs.h"
#include "ppmInput.h"

#define PPMREADBUFLEN 256

int ReadPPM(char * filename,struct Image * pic,char read_only_header)
{
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        char buf[PPMREADBUFLEN], *t;
        unsigned int w=0, h=0, d=0;
        int r=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0) { fclose(pf); return 0; }

        if ( strncmp(buf, "P5\n", 3)  == 0 )
        {
           fprintf(stderr,"Grayscale\n");
           pic->channels=1;
        } else
        if ( strncmp(buf, "P6\n", 3)  == 0 )
        {
           fprintf(stderr,"RGB\n");
           pic->channels=3;
        } else
        { fclose(pf); return 0; }

        do
        { /* Px formats can have # comments after first line */
           t = fgets(buf, PPMREADBUFLEN, pf);
           if ( t == 0 ) { fclose(pf); return 0; }
        } while ( strncmp(buf, "#", 1) == 0 );
        r = sscanf(buf, "%u %u", &w, &h);
        if ( r < 2 ) { fclose(pf); return 0; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if ( (r < 1) /*|| ( d != 255 )*/ ) { fclose(pf); return 0; }

        pic->width=w;
        pic->height=h;
        pic->bitsperpixel=8;

        if ( d <= 255 ) { pic->bitsperpixel=8; } else
        if ( d <= 65535 ) { pic->bitsperpixel=16; }

        fprintf(stderr,"BitsPerPixel %u , Channels %u\n", pic->bitsperpixel,pic->channels);


        if (read_only_header) { fclose(pf); return 1; }

      #if READ_CREATES_A_NEW_PIXEL_BUFFER
	    pic->image_size = pic->height * pic->width * pic->channels * (pic->bitsperpixel/8);
	    pic->pixels = (unsigned char * ) malloc(pic->image_size);
	  #endif


        if ( pic->pixels != 0 )
        {
            size_t rd = fread(pic->pixels,1, pic->image_size, pf);
            fclose(pf);
            if ( rd < pic->image_size )
            {
                fprintf(stderr,"Incorrect read @ file %s , wanted to read %u bytes we got %u ",filename,pic->image_size,rd);
               return 0;
            }
            return 1;
        }
        fclose(pf);
    }
  return 0;
}




int WritePPMOld(char * filename,struct Image * pic)
{

    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (fd!=0)
	{
     unsigned int n=0;

     fprintf(fd, "P6\n%d %d\n255\n", pic->width, pic->height);
     n = (unsigned int ) ( pic->width * pic->height ) ;

     fwrite(pic->pixels, 3, n, fd);

     fflush(fd);
     fclose(fd);

     return 1;
	}

  return 0;
}




int WritePPM(char * filename,struct Image * pic)
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








