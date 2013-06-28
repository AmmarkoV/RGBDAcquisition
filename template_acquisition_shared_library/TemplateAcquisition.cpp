#include "TemplateAcquisition.h"
#include "../acquisition/Acquisition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SAFEGUARD_VALUE 123123
#define MAX_TEMPLATE_DEVICES 5
#define MAX_DIR_PATH 1024
#define PPMREADBUFLEN 256
#define MAX_LINE_CALIBRATION 1024

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052


#define PRINT_COMMENTS 1

struct TemplateVirtualDevice
{
 char readFromDir[MAX_DIR_PATH]; // <- this sucks i know :P
 unsigned int cycle;
 unsigned int safeGUARD;


 unsigned int templateWIDTH;
 unsigned int templateHEIGHT;
 char * templateColorFrame;
 short * templateDepthFrame;

 struct calibration calibRGB;
 struct calibration calibDepth;

};

struct TemplateVirtualDevice device[MAX_TEMPLATE_DEVICES]={0};





int FileExists(char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */
            fclose(fp);
            return 1;
          }
 /* doesnt exist */
 return 0;
}



double getTemplateColorPixelSize(int devID)   { return DEFAULT_PIXEL_SIZE; }
double getTemplateColorFocalLength(int devID)
{
  if (device[devID].calibRGB.intrinsicParametersSet) { return (double) device[devID].calibRGB.intrinsic[0]*getTemplateColorPixelSize(devID); }
  return DEFAULT_FOCAL_LENGTH;
}

double getTemplateDepthFocalLength(int devID)
{ if (device[devID].calibDepth.intrinsicParametersSet) { return (double) device[devID].calibDepth.intrinsic[0]*getTemplateColorPixelSize(devID); }
  return DEFAULT_FOCAL_LENGTH;
}
double getTemplateDepthPixelSize(int devID) { return DEFAULT_PIXEL_SIZE; }


int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib)
{
  calib->intrinsicParametersSet=0;
  calib->extrinsicParametersSet=0;

  calib->intrinsic[0]=0.0;  calib->intrinsic[1]=0.0;  calib->intrinsic[2]=0.0;
  calib->intrinsic[3]=0.0;  calib->intrinsic[4]=0.0;  calib->intrinsic[5]=0.0;
  calib->intrinsic[6]=0.0;  calib->intrinsic[7]=0.0;  calib->intrinsic[8]=0.0;

  float * fx = &calib->intrinsic[0]; float * fy = &calib->intrinsic[4];
  float * cx = &calib->intrinsic[2]; float * cy = &calib->intrinsic[5];
  float * one = &calib->intrinsic[8];

  calib->k1=0.0;  calib->k2=0.0; calib->p1=0.0; calib->p2=0.0; calib->k3=0.0;

  calib->extrinsicRotationRodriguez[0]=0.0; calib->extrinsicRotationRodriguez[1]=0.0; calib->extrinsicRotationRodriguez[2]=0.0;
  calib->extrinsicTranslation[0]=0.0; calib->extrinsicTranslation[1]=0.0; calib->extrinsicTranslation[2]=0.0;

  *cx = (float) width/2;
  *cy = (float) height/2;

  //-This is a bad initial estimation i guess :P
  *fx = (float) DEFAULT_FOCAL_LENGTH/(2*DEFAULT_PIXEL_SIZE);    //<- these might be wrong
  *fy = (float) DEFAULT_FOCAL_LENGTH/(2*DEFAULT_PIXEL_SIZE);    //<- these might be wrong
  //--------------------------------------------

  return 1;
}

int ReadCalibration(char * filename,struct calibration * calib)
{
  FILE * fp = 0;
  fp = fopen(filename,"r");
  if (fp == 0 ) {  return 0; }

  char line[MAX_LINE_CALIBRATION]={0};
  unsigned int lineLength=0;

  unsigned int i=0;

  unsigned int category=0;
  unsigned int linesAtCurrentCategory=0;


  while ( fgets(line,MAX_LINE_CALIBRATION,fp)!=0 )
   {
     unsigned int lineLength = strlen ( line );
     if ( lineLength > 0 ) {
                                 if (line[lineLength-1]==10) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                                 if (line[lineLength-1]==13) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                           }
     if ( lineLength > 1 ) {
                                 if (line[lineLength-2]==10) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                                 if (line[lineLength-2]==13) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                           }


     if (line[0]=='%') { linesAtCurrentCategory=0; }
     if ( (line[0]=='%') && (line[1]=='I') && (line[2]==0) ) { category=1;    } else
     if ( (line[0]=='%') && (line[1]=='D') && (line[2]==0) ) { category=2;    } else
     if ( (line[0]=='%') && (line[1]=='T') && (line[2]==0) ) { category=3;    } else
     if ( (line[0]=='%') && (line[1]=='R') && (line[2]==0) ) { category=4;    } else
        {
          fprintf(stderr,"Line %u ( %s ) is category %u lines %u \n",i,line,category,linesAtCurrentCategory);
          if (category==1)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->intrinsic[0] = atof(line); break;
             case 2 :  calib->intrinsic[1] = atof(line); break;
             case 3 :  calib->intrinsic[2] = atof(line); break;
             case 4 :  calib->intrinsic[3] = atof(line); break;
             case 5 :  calib->intrinsic[4] = atof(line); break;
             case 6 :  calib->intrinsic[5] = atof(line); break;
             case 7 :  calib->intrinsic[6] = atof(line); break;
             case 8 :  calib->intrinsic[7] = atof(line); break;
             case 9 :  calib->intrinsic[8] = atof(line); break;
           };
          } else
          if (category==2)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->k1 = atof(line); break;
             case 2 :  calib->k2 = atof(line); break;
             case 3 :  calib->p1 = atof(line); break;
             case 4 :  calib->p2 = atof(line); break;
             case 5 :  calib->k3 = atof(line); break;
           };
          } else
          if (category==3)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicTranslation[0] = atof(line); break;
             case 2 :  calib->extrinsicTranslation[1] = atof(line); break;
             case 3 :  calib->extrinsicTranslation[2] = atof(line); break;
           };
          } else
          if (category==4)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicRotationRodriguez[0] = atof(line); break;
             case 2 :  calib->extrinsicRotationRodriguez[1] = atof(line); break;
             case 3 :  calib->extrinsicRotationRodriguez[2] = atof(line); break;
           };
          }


        }

     ++linesAtCurrentCategory;
     ++i;
     line[0]=0;
   }

  return 1;
}


int getTemplateColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &device[devID].calibRGB,sizeof(struct calibration));
    return 1;
}

int getTemplateDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &device[devID].calibDepth,sizeof(struct calibration));
    return 1;
}



char * ReadPPM(char * filename,unsigned int *width,unsigned int *height)
{
    fprintf(stderr,"Reading template file %s \n",filename);
    char * pixels=0;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        char buf[PPMREADBUFLEN], *t;
        unsigned int w=0, h=0, d=0;
        int r=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if ( (t == 0) || ( strncmp(buf, "P6\n", 3) != 0 ) ) { fprintf(stderr,"ReadPPM only undertsands P6 format\n"); fclose(pf); return 0; }
        do
        { /* Px formats can have # comments after first line */
           #if PRINT_COMMENTS
             memset(buf,0,PPMREADBUFLEN);
           #endif
           t = fgets(buf, PPMREADBUFLEN, pf);
           if ( t == 0 ) { fclose(pf); return 0; }
           #if PRINT_COMMENTS
             if (buf[0]=='#') { printf("%s\n",buf+1); } //<- Printout Comment!
           #endif
        } while ( strncmp(buf, "#", 1) == 0 );
        r = sscanf(buf, "%u %u", &w, &h);
        if ( r < 2 ) { fclose(pf); fprintf(stderr,"Incoherent dimensions received %ux%u \n",w,h); return 0; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if ( (r < 1) || ( d != 255 ) ) { fprintf(stderr,"Incoherent payload received %u bits per pixel \n",d); fclose(pf); return 0; }


        *width=w;
        *height=h;
        pixels= (char*) malloc(w*h*3*sizeof(char));

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,3, w*h, pf);
          if (rd < w*h )
             {
               fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd, w*h);
             }

          fclose(pf);
          //if ( rd < w*h ) { return 0; }
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

  return 0;
}




short * ReadPPMD(char * filename,unsigned int *width,unsigned int *height)
{
    fprintf(stderr,"Reading template file %s \n",filename);
    short * pixels=0;
    FILE *pf=0;
    pf = fopen(filename,"rb");

    if (pf!=0 )
    {
        char buf[PPMREADBUFLEN], *t;
        unsigned int w=0, h=0, d=0;
        int r=0;

        t = fgets(buf, PPMREADBUFLEN, pf);
        if ( (t == 0) || ( strncmp(buf, "P5\n", 3) != 0 ) ) { fclose(pf); return 0; }
        do
        { /* Px formats can have # comments after first line */
           #if PRINT_COMMENTS
             memset(buf,0,PPMREADBUFLEN);
           #endif
           t = fgets(buf, PPMREADBUFLEN, pf);
           if ( t == 0 ) { fclose(pf); return 0; }
           #if PRINT_COMMENTS
             if (buf[0]=='#') { printf("%s\n",buf+1); } //<- Printout Comment!
           #endif
        } while ( strncmp(buf, "#", 1) == 0 );
        r = sscanf(buf, "%u %u", &w, &h);
        if ( r < 2 ) { fclose(pf); return 0; }
        // The program fails if the first byte of the image is equal to 32. because
        // the fscanf eats the space and the image is read with some bit less
        r = fscanf(pf, "%u\n", &d);
        if ( (r < 1) /*|| ( d != 255 )*/ ) { fclose(pf); return 0; }


        *width=w;
        *height=h;
        pixels= (short*) malloc(w*h*3*sizeof(short));

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,sizeof(short), w*h, pf);
          if (rd < w*h) { fprintf(stderr,"Note : Incomplete read while reading file %s (%u instead of %u)\n",filename,(unsigned int) rd,w*h);  }

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

  return 0;
}



int startTemplate(unsigned int max_devs,char * settings)
{
    unsigned int devID = 0;
    for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
    {
        //fprintf(stderr,"Zeroing device %u\n",devID);
        device[devID].templateWIDTH = 640;
        device[devID].templateHEIGHT = 480;

        device[devID].readFromDir[0]=0; // <- this sucks i know :P
        device[devID].cycle=0;

        device[devID].safeGUARD = SAFEGUARD_VALUE;

        device[devID].templateColorFrame=0;
        device[devID].templateDepthFrame=0;
    }

    fprintf(stderr,"startTemplate done \n");
    return 1;
}




int getTemplateNumberOfDevices() { return 1; }

int stopTemplate()
{
   unsigned int devID = 0;
   for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
   {
     if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); device[devID].templateColorFrame=0; }
     if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); device[devID].templateDepthFrame=0; }
   }

   return 1;
}

int createTemplateDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  if ( ( device[devID].templateWIDTH < width ) &&  ( device[devID].templateHEIGHT < height ) )
   {
        device[devID].templateHEIGHT=height;
        device[devID].templateWIDTH=width;
   }

   if (devName==0) { strcpy(device[devID].readFromDir,""); } else
     {
       if (strlen(devName)==0)  { strcpy(device[devID].readFromDir,""); } else
                                { strncpy(device[devID].readFromDir,devName,MAX_DIR_PATH);  }
     }

  unsigned int widthInternal; unsigned int heightInternal;

  char file_name_test[1024];
  sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devID,0);
  char * tmpColor = ReadPPM(file_name_test,&widthInternal,&heightInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
   { fprintf(stderr,"Please note that the templateColor.pnm file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",widthInternal,heightInternal,width,height); }

  if (tmpColor!=0) { device[devID].templateColorFrame=tmpColor; } else
  {
   // if templateColorFrame is zero the next function behaves like a malloc
   device[devID].templateColorFrame= (char*) realloc(device[devID].templateColorFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*3*sizeof(char));
  }


  sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devID,0);
  short * tmpDepth = ReadPPMD(file_name_test,&widthInternal,&heightInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
   { fprintf(stderr,"Please note that the templateColor.pnm file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",widthInternal,heightInternal,width,height); }

  if (tmpDepth!=0) { device[devID].templateDepthFrame=tmpDepth; } else
  {
   // if templateDepthFrame is zero the next function behaves like a malloc
   device[devID].templateDepthFrame= (short*) realloc(device[devID].templateDepthFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*1*sizeof(short));
  }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibRGB);

  sprintf(file_name_test,"frames/%s/color.calib",device[devID].readFromDir);
  if ( ! ReadCalibration(file_name_test,&device[devID].calibRGB) ) { fprintf(stderr,"Could not read color calibration\n"); }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibDepth);

  sprintf(file_name_test,"frames/%s/depth.calib",device[devID].readFromDir);
  if ( ! ReadCalibration(file_name_test,&device[devID].calibDepth) ) { fprintf(stderr,"Could not read depth calibration\n"); }


  return ((device[devID].templateColorFrame!=0)&&(device[devID].templateDepthFrame!=0));
}



int destroyTemplateDevice(int devID)
{
  return 1;
}


int seekTemplateFrame(int devID,unsigned int seekFrame)
{
  device[devID].cycle = seekFrame;
  return 1;
}

int snapTemplateFrames(int devID)
{
    //TODO HERE MAYBE LOAD NEW BUFFERS
    int found_frames = 0;

    unsigned int widthInternal=0; unsigned int heightInternal=0;
    char * file_name_test = (char* ) malloc(2048 * sizeof(char));
    if (file_name_test==0) { fprintf(stderr,"Could not snap frame , no space for string\n"); return 0; }

    sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devID,device[devID].cycle);
    //fprintf(stderr,"Snap color %s",file_name_test);
    if (FileExists(file_name_test))
     {
       if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); }
       device[devID].templateColorFrame = ReadPPM(file_name_test,&widthInternal,&heightInternal);
       ++found_frames;
     }

    sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devID,device[devID].cycle);
    //fprintf(stderr,"Snap depth %s",file_name_test);
    if (FileExists(file_name_test))
     {
      if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); }
      device[devID].templateDepthFrame = ReadPPMD(file_name_test,&widthInternal,&heightInternal);
      ++found_frames;
     }

  free(file_name_test);
  file_name_test=0;

  ++device[devID].cycle;
  if ( device[devID].safeGUARD != SAFEGUARD_VALUE ) { fprintf(stderr,"\n\n\n\nERROR , memory corruption \n\n\n\n"); }

  if (device[devID].cycle>65534) { device[devID].cycle=0; }
  if (found_frames==0) { fprintf(stderr,"Could not find any frames , we finished stream \n");  device[devID].cycle = 0; } else
  if (found_frames!=2) { fprintf(stderr,"\n Warning: Did not find both frames\n");   }

  return 1;
}

//Color Frame getters
int getTemplateColorWidth(int devID)        { return device[devID].templateWIDTH; }
int getTemplateColorHeight(int devID)       { return device[devID].templateHEIGHT; }
int getTemplateColorDataSize(int devID)     { return device[devID].templateHEIGHT*device[devID].templateWIDTH * 3; }
int getTemplateColorChannels(int devID)     { return 3; }
int getTemplateColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
char * getTemplateColorPixels(int devID)    { return device[devID].templateColorFrame; }




   //Depth Frame getters
int getTemplateDepthWidth(int devID)    { return device[devID].templateWIDTH; }
int getTemplateDepthHeight(int devID)   { return device[devID].templateHEIGHT; }
int getTemplateDepthDataSize(int devID) { return device[devID].templateWIDTH*device[devID].templateHEIGHT; }
int getTemplateDepthChannels(int devID)     { return 1; }
int getTemplateDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getTemplateDepthPixels(int devID) { return (char *) device[devID].templateDepthFrame; }

