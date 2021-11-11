
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pfmInput.h"

#define PRINT_COMMENTS 1
#define PPMREADBUFLEN 256

float * ReadPFMRaw(float * buffer , char * filename, unsigned int *width , unsigned int *height ,   unsigned int * bytesPerPixel , unsigned int * channels)
{
   * bytesPerPixel = 0;
   * channels = 0;

    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    float * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0; *height=0;
        char buf[PPMREADBUFLEN]={0};
        char *t;
        unsigned int w=0, h=0, d=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0) { fclose(pf); return buffer; }

        if ( strncmp(buf,"PF\n", 3) == 0 ) { *channels=3; } else
        if ( strncmp(buf,"Pf\n", 3) == 0 ) { *channels=1; } else
                                           { fprintf(stderr,"Could not understand/Not supported file format\n"); fclose(pf); return buffer; }
        int z = sscanf(buf, "%u %u", &w, &h);
        if ( z < 2 ) { fclose(pf); fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h); return buffer; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        fscanf(pf, "%u\n", &d);

        *bytesPerPixel=1; // This is fixed


        //This is a super ninja hackish patch that fixes the case where fscanf eats one character more on the stream
        //It could be done better  ( no need to fseek ) but this will have to do for now
        //Scan for border case
           unsigned long startOfBinaryPart = ftell(pf);
           if ( fseek (pf , 0 , SEEK_END)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
           unsigned long totalFileSize = ftell (pf); //lSize now holds the size of the file..

           //fprintf(stderr,"totalFileSize-startOfBinaryPart = %u \n",totalFileSize-startOfBinaryPart);
           //fprintf(stderr,"bytesPerPixel*channels*w*h = %u \n",bytesPerPixel*channels*w*h);
           if (totalFileSize-startOfBinaryPart < *bytesPerPixel*(*channels)*w*h )
           {
              fprintf(stderr," Detected Border Case\n\n\n");
              startOfBinaryPart-=1;
           }
           if ( fseek (pf , startOfBinaryPart , SEEK_SET)!=0 ) { fprintf(stderr,"Could not find file size to cache client..!\nUnable to serve client\n"); fclose(pf); return 0; }
         //-----------------------
         //----------------------

        *width=w; *height=h;
        if (pixels==0) {  pixels= (float *) malloc(w*h*(*bytesPerPixel)*(*channels)*sizeof(float )); }

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,*bytesPerPixel*(*channels), w*h, pf);
          if (rd < w*h )
             {
               fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd, w*h);
               fprintf(stderr,"Dimensions ( %u x %u ) , Depth %u bytes , Channels %u \n",w,h,*bytesPerPixel,*channels);
             }

          fclose(pf);


          return pixels;
        } else
        {
            fprintf(stderr,"Could not Allocate enough memory for file %s \n",filename);
        }
        fclose(pf);
    } else
    {
      fprintf(stderr,"File %s does not exist \n",filename);
    }
  return buffer;
}



int ReadPFM(char * filename,struct Image * pic,char read_only_header)
{
  pic->timestamp = 0;
  pic->pixels = (unsigned char*) ReadPFMRaw((float*) pic->pixels , filename, &pic->width, &pic->height ,&pic->bitsperpixel , &pic->channels );
  pic->bitsperpixel = pic->bitsperpixel * 8; // ( we go from bytes to bits )

  return (pic->pixels!=0);
}

