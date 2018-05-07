#ifndef DRAWING_H_INCLUDED
#define DRAWING_H_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


static unsigned int simplePowCodecsDrawing(unsigned int base,unsigned int exp)
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


static int writePPMDrawing(const char * filename,const char * pixels , unsigned int width , unsigned int height, unsigned int channels, unsigned int bitsperpixel)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called bitsperpixel=%u \n",filename,bitsperpixel);

    //fflush(stderr);
    if (pixels==0) { return 0; }
    if ( (width==0) || (height==0) || (channels==0) || (bitsperpixel==0) )
        {
          fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions ( %ux%u %u channels %u bpp\n",filename,width , height,channels,bitsperpixel);
          return 0;
        }
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel ( requested %u )..!\n",bitsperpixel); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");

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

        fprintf(fd, "%d %d\n%u\n", width, height , simplePowCodecsDrawing(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

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

static int blitRGB( char * target , unsigned int targetWidth , unsigned int targetHeight ,
                     unsigned char r , unsigned char g , unsigned b,
                     unsigned int xS,unsigned int yS, unsigned int width , unsigned int height)
{
  //fprintf(stderr,"blitRGB(%u,%u,%u,%u)\n",xS,yS,width,height);

  if (xS>=targetWidth)  { fprintf(stderr,"E1"); xS=targetWidth-1;   }
  if (yS>=targetHeight) { fprintf(stderr,"E2"); yS=targetHeight-1;  }

  if (xS+width>=targetWidth)   { fprintf(stderr,"E3"); width=targetWidth-xS-1;    }
  if (yS+height>=targetHeight) { fprintf(stderr,"E4"); height=targetHeight-yS-1;  }

  unsigned int x,y;
  for (y=yS; y<yS+height; y++)
  {
    for (x=xS; x<xS+width; x++)
    {
      //fprintf(stderr,"(%u,%u)\n",x,y);
      char * ptr = target + ((x*3) + (y*targetWidth*3));

      *ptr = r; ptr++;
      *ptr = g; ptr++;
      *ptr = b;
    }
  }
 return 1;
}



static int drawRectangleRGB(char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                     unsigned char r , unsigned char g , unsigned char b,unsigned int thickness,
                     unsigned int x1,unsigned int y1, unsigned int x2, unsigned int y2)
{
  //fprintf(stderr,"drawRectangleRGB(%u,%u,%u,%u) thickness=%u image=(%ux%u) \n",x1,y1,x2,y2,thickness , targetWidth,targetHeight);
  //Make sure x1,y1, x2,y2 are ordered correctly
  unsigned int tmp;
  if (x1>x2)  { tmp=x2; x2=x1; x1=tmp; }
  if (y1>y2)  { tmp=y2; y2=y1; y1=tmp; }


  //Make sure x1,y1, x2,y2 have enough space for our thickness
  if (x1+thickness>=targetWidth)  { x1=targetWidth-thickness-1;  }
  if (y1+thickness>=targetHeight) { y1=targetHeight-thickness-1;  }
  if (x2<thickness)               { x2=thickness;  }
  if (y2<thickness)               { y2=thickness;  }


  //Calculate width/height
  unsigned int width = x2-x1;
  unsigned int height = y2-y1;

  //Blit in the rectangle
  blitRGB(target,targetWidth,targetHeight, r,g ,b,  x1 ,y1 , width , thickness);              //top
  blitRGB(target,targetWidth,targetHeight, r,g ,b,  x1 ,y2-thickness , width , thickness);    //bottom
  blitRGB(target,targetWidth,targetHeight, r,g ,b,  x1 ,y1 , thickness , height );            //left
  blitRGB(target,targetWidth,targetHeight, r,g ,b,  x2-thickness , y1 , thickness, height);   //right
  return 1;
}



#ifdef __cplusplus
}
#endif

#endif // DRAWING_H_INCLUDED
