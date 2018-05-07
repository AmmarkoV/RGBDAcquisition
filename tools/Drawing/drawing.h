#ifndef DRAWING_H_INCLUDED
#define DRAWING_H_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define PPMREADBUFLEN 256

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


static unsigned char * ReadPNMDrawing(unsigned char * buffer , char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp , unsigned int * bytesPerPixel , unsigned int * channels)
{
   * bytesPerPixel = 0;
   * channels = 0;

   if (timestamp!=0) { *timestamp=0; }

    //See http://en.wikipedia.org/wiki/Portable_anymap#File_format_description for this simple and useful format
    unsigned char * pixels=buffer;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        *width=0; *height=0;

        char buf[PPMREADBUFLEN]={0};
        char *t;
        unsigned int w=0, h=0, d=0;
        int r=0 , z=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if (t == 0) { return buffer; }

        if ( strncmp(buf,"P6\n", 3) == 0 ) { *channels=3; } else
        if ( strncmp(buf,"P5\n", 3) == 0 ) { *channels=1; } else
                                           { fprintf(stderr,"Could not understand/Not supported file format\n"); fclose(pf); return buffer; }
        do
        { /* Px formats can have # comments after first line */
           #if PRINT_COMMENTS
             memset(buf,0,PPMREADBUFLEN);
           #endif
           t = fgets(buf, PPMREADBUFLEN, pf);

           if (timestamp!=0)
           {
            if (strstr(buf,"TIMESTAMP")!=0)
              {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
              }
           }

           if ( t == 0 ) { fclose(pf); return buffer; }
        } while ( strncmp(buf, "#", 1) == 0 );
        z = sscanf(buf, "%u %u", &w, &h);
        if ( z < 2 ) { fclose(pf); fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h); return buffer; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if (r < 1) { fprintf(stderr,"Could not understand how many bytesPerPixel there are on this image\n"); fclose(pf); return buffer; }
        if (d==255) { *bytesPerPixel=1; }  else
        if (d==65535) { *bytesPerPixel=2; } else
                       { fprintf(stderr,"Incoherent payload received %u bits per pixel \n",d); fclose(pf); return buffer; }


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
        if (pixels==0) {  pixels= (unsigned char*) malloc(w*h*(*bytesPerPixel)*(*channels)*sizeof(char)); }

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,*bytesPerPixel*(*channels), w*h, pf);
          if (rd < w*h )
             {
               fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd, w*h);
               fprintf(stderr,"Dimensions ( %u x %u ) , Depth %u bytes , Channels %u \n",w,h,*bytesPerPixel,*channels);
             }

          fclose(pf);

           #if PRINT_COMMENTS
             if ( (*channels==1) && (*bytesPerPixel==2) && (timestamp!=0) ) { printf("DEPTH %lu\n",*timestamp); } else
             if ( (*channels==3) && (*bytesPerPixel==1) && (timestamp!=0) ) { printf("COLOR %lu\n",*timestamp); }
           #endif

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


static int writePPMDrawing(const char * filename,const unsigned char * pixels , unsigned int width , unsigned int height, unsigned int channels, unsigned int bitsperpixel)
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
