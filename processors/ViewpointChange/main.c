#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ViewpointChange.h"
#include "viewpoint_change.h"




#define PPMREADBUFLEN 256

unsigned char * viewPointChange_ReadPPM(char * filename,unsigned int * width , unsigned int * height , unsigned int * channels,unsigned int * bitsperpixel,char read_only_header)
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
           *channels=1;
        } else
        if ( strncmp(buf, "P6\n", 3)  == 0 )
        {
           fprintf(stderr,"RGB\n");
           *channels=3;
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

        *width=w;
        *height=h;
        *bitsperpixel=8;

        if ( d <= 255 ) { *bitsperpixel=8; } else
        if ( d <= 65535 ) { *bitsperpixel=16; }

        fprintf(stderr,"BitsPerPixel %u , Channels %u\n", *bitsperpixel,*channels);


        if (read_only_header) { fclose(pf); return 0; }

	    unsigned int pic_image_size = (*height) * (*width) * (*channels) * ((unsigned int) (*bitsperpixel)/8);
	    unsigned char * pic_pixels = (unsigned char * ) malloc(pic_image_size);

        if ( pic_pixels  != 0 )
        {
            size_t rd = fread(pic_pixels ,1, pic_image_size , pf);
            fclose(pf);
            if ( rd < pic_image_size )
            {
                fprintf(stderr,"Incorrect read @ file %s , wanted to read %u bytes we got %u ",filename,pic_image_size,rd);
               return pic_pixels;
            }
            return pic_pixels;
        }
        fclose(pf);
    }
  return 0;
}





unsigned int viewPointChange_fitImageInMask(unsigned char * img, unsigned char * mask , unsigned int width , unsigned int height)
{
  if ( (img==0)||(mask==0) ) { fprintf(stderr,"Cannot FitImageInMask with empty Images\n"); return 0; }

  unsigned char * imgPtr = img;
  unsigned char * imgLimit = imgPtr + (width * height * 3);
  unsigned char * maskPtr = mask;

  unsigned int thisPixelCounts = 0;
  unsigned int countPositive = 0;
  unsigned int countNegative = 0;

  while (imgPtr < imgLimit)
  {
      if ((*maskPtr)==0)
      { imgPtr+=3; } else
      if ((*maskPtr)==123) //Masked Values 123 x x are Negatives
      {
        thisPixelCounts = 0;
        if (*imgPtr==0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr==0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr==0) { thisPixelCounts=1; } ++imgPtr;
        if (thisPixelCounts!=0) { ++countPositive; }
      } else
      if ((*maskPtr)!=0) //Masked Values 255 x x are Positives
      {
        thisPixelCounts = 0;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (thisPixelCounts!=0) { ++countNegative; }
      } else
      { imgPtr+=3; }

     maskPtr+=3;
  }

   fprintf(stderr,"Positive Count %u , Negative Count %u \n",countPositive,countNegative);


   return countPositive+countNegative;
}



unsigned char* viewPointChange_mallocTransformToBirdEyeView(unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height , unsigned int depthRange)
{
  return  birdsEyeView(rgb , depth , width, height , 0 , depthRange);
}


int viewPointChange_newFramesCompare(unsigned char *  prototype , unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height  , unsigned int depthRange )
{
  unsigned char * bev = birdsEyeView(rgb , depth , width, height, 0, depthRange);
     if (bev!=0)
         {
           unsigned int fitScore = viewPointChange_fitImageInMask(bev,rgb,width,height);
           fprintf(stderr,"Got a fit of %u\n",fitScore);

           if (fitScore < 4000)
           {
             fprintf(stderr,"\n");
           }

           free(bev);
        } else
        { fprintf(stderr,"Could not perform a birdseyeview translation\n"); }

  return 1;
}





unsigned int viewPointChange_countDepths(unsigned short *  depth, unsigned int imageWidth , unsigned int imageHeight  ,
                                unsigned int x , unsigned int y , unsigned int width , unsigned int height ,
                                unsigned int depthRange )
{
  unsigned int depthSamples = 0;
  unsigned int totalDepth = 0;

  unsigned short * depthPTR;
  unsigned short * depthStart=depth+ (y * imageWidth) +x;
  unsigned short * depthLimit = depthStart + width;
  unsigned short * depthTotalLimit = depthLimit + imageWidth*height;

  while (depthPTR<depthTotalLimit)
  {
   depthPTR = depthStart;
   while ( depthPTR < depthLimit )
   {
     totalDepth+=*depthPTR;
     ++depthSamples;

    ++depthPTR;
   }
   depthStart+=imageWidth;
   depthLimit+=imageWidth;
  }

  if (depthSamples==0) { depthSamples=1; }
  return totalDepth/depthSamples;
}

