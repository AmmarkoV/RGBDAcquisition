#include "TemplateAcquisition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

unsigned int templateWIDTH=640;
unsigned int templateHEIGHT=480;
char * templateColorFrame = 0;
short * templateDepthFrame = 0;

unsigned short cycle=0;


#define PPMREADBUFLEN 256

char * ReadPPM(char * filename,unsigned int *width,unsigned int *height)
{
    char * pixels=0;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        char buf[PPMREADBUFLEN], *t;
        unsigned int w=0, h=0, d=0;
        int r=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if ( (t == 0) || ( strncmp(buf, "P6\n", 3) != 0 ) ) { fclose(pf); return 0; }
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
        if ( (r < 1) || ( d != 255 ) ) { fclose(pf); return 0; }


        *width=w;
        *height=h;

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,3, w*h, pf);
          fclose(pf);
          if ( rd < w*h ) { return 0; }
          return pixels;
        }
        fclose(pf);
    }
  return 0;
}








int startTemplate(unsigned int max_devs) { return 1; }
int getTemplateNumberOfDevices() { return 1; }

int stopTemplate()
{
   if (templateColorFrame!=0) { free(templateColorFrame); templateColorFrame=0; }
   if (templateDepthFrame!=0) { free(templateDepthFrame); templateDepthFrame=0; }
   return 1;
}

int createTemplateDevice(int devID,unsigned int width,unsigned int height,unsigned int framerate)
{
  if ( ( templateWIDTH < width ) &&  ( templateHEIGHT < height ) )
   {
        templateHEIGHT=height;
        templateWIDTH=width;
   }

  // if templateColorFrame is zero the next function behaves like a malloc
  templateColorFrame= (char*) realloc(templateColorFrame,templateWIDTH*templateHEIGHT*3*sizeof(char));
  // if templateColorFrame is zero the next function behaves like a malloc
  templateDepthFrame= (short*) realloc(templateDepthFrame,templateWIDTH*templateHEIGHT*1*sizeof(short));

  return ((templateColorFrame!=0)&&(templateDepthFrame!=0));
}



int destroyTemplateDevice(int devID)
{
  return 1;
}


int snapTemplateFrames(int devID)
{
  ++cycle;
  if (cycle>65534) { cycle=0; }

  unsigned int rgb_cycle = (unsigned int ) (65535-cycle)/255;
  unsigned int i =0 ;
  unsigned int j =0 ;
  while ( i < templateWIDTH*templateHEIGHT*3 )
    {
        templateColorFrame[i]=rgb_cycle;
        templateColorFrame[i+1]=rgb_cycle;
        templateColorFrame[i+2]=rgb_cycle;
        i+=3;

        templateDepthFrame[j]=cycle;
        ++j;
    }
  return 1;
}

//Color Frame getters
int getTemplateColorWidth(int devID)        { return templateWIDTH; }
int getTemplateColorHeight(int devID)       { return templateHEIGHT; }
int getTemplateColorDataSize(int devID)     { return templateHEIGHT*templateWIDTH * 3; }
int getTemplateColorChannels(int devID)     { return 3; }
int getTemplateColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
char * getTemplateColorPixels(int devID)    { return templateColorFrame; }

double getTemplateColorFocalLength(int devID) { return 120.0; }
double getTemplateColorPixelSize(int devID)   { return 0.1052; }


   //Depth Frame getters
int getTemplateDepthWidth(int devID)    { return templateWIDTH; }
int getTemplateDepthHeight(int devID)   { return templateHEIGHT; }
int getTemplateDepthDataSize(int devID) { return templateWIDTH*templateHEIGHT; }
int getTemplateDepthChannels(int devID)     { return 1; }
int getTemplateDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getTemplateDepthPixels(int devID) { return (char *) templateDepthFrame; }

double getTemplateDepthFocalLength(int devID) { return 120.0; }
double getTemplateDepthPixelSize(int devID) { return 0.1052; }

