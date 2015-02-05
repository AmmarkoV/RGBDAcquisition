#include "imageFilters.h"

#include <stdio.h>
#include <stdlib.h>

typedef unsigned char BYTE;


struct Image
{
  unsigned char * pixels;
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  unsigned int image_size;
};


int monochrome(struct Image * img)
{
 if (img==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->pixels==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->channels!=3) { fprintf(stderr,"Function Monochrome assumes 3byte array\n"); return 0; }

 if (img->channels==1) { /*Already Monochrome */return 1; }

 BYTE * input_frame = img->pixels;
 unsigned int col_med;
 unsigned int image_size= img->width * img->height * img->channels * (img->bitsperpixel/8);

 register BYTE *out_px = (BYTE *) input_frame;
 register BYTE *end_px = (BYTE *) input_frame + image_size;
 register BYTE *px = (BYTE *) input_frame;
 register BYTE *r;
 register BYTE *g;
 register BYTE *b;

 while ( px < end_px)
 {
       r = px++; g = px++; b = px++;
       col_med= ( *r + *g + *b )/3;
       *out_px= (BYTE)col_med ;
       ++out_px;
 }
 img->channels = 1;
 return 1;
}



int contrast(struct Image * img,float scaleValue)
{
 if (img==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->pixels==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->channels!=3) { fprintf(stderr,"Function Monochrome assumes 3byte array\n"); return 0; }

 if (img->channels==1) { /*Already Monochrome */return 1; }

 BYTE * input_frame = img->pixels;
 unsigned int col_med;
 unsigned int image_size= img->width * img->height * img->channels * (img->bitsperpixel/8);

 register BYTE *out_px = (BYTE *) input_frame;
 register BYTE *end_px = (BYTE *) input_frame + image_size;
 register BYTE *px = (BYTE *) input_frame;
 register BYTE *r;
 register BYTE *g;
 register BYTE *b;

 unsigned int rScaled,gScaled,bScaled;

 while ( px < end_px)
 {
       r = px++; g = px++; b = px++;

       rScaled = *r;
       rScaled=(unsigned int) (rScaled * scaleValue);
       gScaled = *g;
       gScaled=(unsigned int) (gScaled * scaleValue);
       bScaled = *b;
       bScaled=(unsigned int) (bScaled * scaleValue);

       if (rScaled>255) { *r=255; } else { *r = (unsigned char) rScaled; }
       if (gScaled>255) { *g=255; } else { *g = (unsigned char) gScaled; }
       if (bScaled>255) { *b=255; } else { *b = (unsigned char) bScaled; }
 }
 return 1;
}
