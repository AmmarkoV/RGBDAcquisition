#include <stdio.h>
#include <stdlib.h>

#include "TemplateAcquisition.h"

#define USE_DIRECTORY_LISTING 1
#define BUILD_TEMPLATE 1

#if BUILD_TEMPLATE
#include "../acquisition/Acquisition.h"


    #if USE_DIRECTORY_LISTING
     #include "../tools/OperatingSystem/OperatingSystem.h"
    #endif // USE_DIRECTORY_LISTING


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
#define PRINT_DEBUG_EACH_CALL 0

struct TemplateVirtualDevice
{
 char readFromDir[MAX_DIR_PATH]; // <- this sucks i know :P
 unsigned int cycle;
 unsigned int totalFrames;
 unsigned int safeGUARD;


 unsigned int templateWIDTH;
 unsigned int templateHEIGHT;
 unsigned long lastColorTimestamp;
 unsigned char * templateColorFrame;
 unsigned long lastDepthTimestamp;
 unsigned short * templateDepthFrame;

 struct calibration calibRGB;
 struct calibration calibDepth;

};

struct TemplateVirtualDevice device[MAX_TEMPLATE_DEVICES]={0};

int makeFrameNoInput(unsigned char * frame , unsigned int width , unsigned int height , unsigned int channels)
{
   unsigned char * framePTR = frame;
   unsigned char * frameLimit = frame + width * height * channels * sizeof(char);

   while (framePTR<frameLimit)
   {
       *framePTR=0; ++framePTR;
       *framePTR=0; ++framePTR;
       *framePTR=255; ++framePTR;
   }
 return 1;
}


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


#if USE_CALIBRATION
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


int setTemplateColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &device[devID].calibRGB , (void*) calib,sizeof(struct calibration));
    return 1;
}

int setTemplateDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &device[devID].calibDepth , (void*) calib,sizeof(struct calibration));
    return 1;
}
#endif



unsigned char * ReadPPM(char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp)
{
    #if PRINT_DEBUG_EACH_CALL
     fprintf(stderr,"TemplateAcquisition : Reading file %s \n",filename);
    #endif // PRINT_DEBUG_EACH_CALL

    unsigned char * pixels=0;
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
           if (strstr(buf,"TIMESTAMP")!=0)
              {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
              }

           if ( t == 0 ) { fclose(pf); return 0; }
           #if PRINT_COMMENTS
             if (buf[0]=='#') { printf("COLOR %s\n",buf+1); } //<- Printout Comment!
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
        pixels= (unsigned char*) malloc(w*h*3*sizeof(unsigned char));

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




unsigned short * ReadPPMD(char * filename,unsigned int *width,unsigned int *height,unsigned long * timestamp)
{
    #if PRINT_DEBUG_EACH_CALL
     fprintf(stderr,"TemplateAcquisition : Reading file %s \n",filename);
    #endif // PRINT_DEBUG_EACH_CALL

    unsigned short * pixels=0;
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

           if (strstr(buf,"TIMESTAMP")!=0)
              {
                char * timestampPayloadStr = buf + 10;
                *timestamp = atoi(timestampPayloadStr);
              }

           if ( t == 0 ) { fclose(pf); return 0; }
           #if PRINT_COMMENTS
             if (buf[0]=='#') { printf("DEPTH %s\n",buf+1); } //<- Printout Comment!
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
        pixels= (unsigned short*) malloc(w*h*sizeof(unsigned short)); /*Only 1 channel in depth images 3*/

        if ( pixels != 0 )
        {
          size_t rd = fread(pixels,sizeof(unsigned short), w*h, pf);
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



int flipDepth(unsigned short * depth,unsigned int width , unsigned int height )
{
  unsigned char tmp ;
  unsigned char * depthPtr=(unsigned char *) depth;
  unsigned char * depthPtrNext=(unsigned char *) depth+1;
  unsigned char * depthPtrLimit =(unsigned char *) depth + width * height * 2 ;
  while ( depthPtr < depthPtrLimit )
  {
     tmp=*depthPtr;
     *depthPtr=*depthPtrNext;
     *depthPtrNext=tmp;

     ++depthPtr;
     ++depthPtrNext;
  }

 return 0;
}



int startTemplateModule(unsigned int max_devs,char * settings)
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

int stopTemplateModule()
{
   unsigned int devID = 0;
   for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
   {
     if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); device[devID].templateColorFrame=0; }
     if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); device[devID].templateDepthFrame=0; }
   }

   return 1;
}


int findLastFrame(int devID)
{
  unsigned int i=0;
  char file_name_test[1024];

  while (i<100000)
  {
   device[devID].totalFrames = i;
   sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devID,i);
   if ( ! FileExists(file_name_test) ) { break; }
   sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devID,i);
   if ( ! FileExists(file_name_test) ) { break; }
   ++i;
  }

  return 1;
}

int listTemplateDevices(int devID,char * output, unsigned int maxOutput)
{
    #if USE_DIRECTORY_LISTING
     char where2Search[]="frames/";
     return listDirectory(where2Search, output, maxOutput);
    #endif // USE_DIRECTORY_LISTING
    return 0;
}


int createTemplateDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 device[devID].cycle=0;
 device[devID].totalFrames=0;
 device[devID].templateWIDTH=0;
 device[devID].templateHEIGHT=0;

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

  findLastFrame(devID);

  unsigned int failedStream=0;
  unsigned int widthInternal=0; unsigned int heightInternal=0; unsigned long timestampInternal=0;

  char file_name_test[1024];
  sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devID,0);
  unsigned char * tmpColor = ReadPPM(file_name_test,&widthInternal,&heightInternal, &timestampInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
   { fprintf(stderr,"Please note that the templateColor.pnm file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",widthInternal,heightInternal,width,height); }

  if (tmpColor!=0) { device[devID].templateColorFrame=tmpColor; } else
  {
   ++failedStream;
   // if templateColorFrame is zero the next function behaves like a malloc
   device[devID].templateColorFrame= (unsigned char*) realloc(device[devID].templateColorFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*3*sizeof(char));
   makeFrameNoInput(device[devID].templateColorFrame,device[devID].templateWIDTH,device[devID].templateHEIGHT,3);
  }


  sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devID,0);
  unsigned short * tmpDepth = ReadPPMD(file_name_test,&widthInternal,&heightInternal, &timestampInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
   { fprintf(stderr,"Please note that the templateColor.pnm file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",widthInternal,heightInternal,width,height); }

  if (tmpDepth!=0) { device[devID].templateDepthFrame=tmpDepth; } else
  {
   ++failedStream;
   // if templateDepthFrame is zero the next function behaves like a malloc
   device[devID].templateDepthFrame= (unsigned short*) realloc(device[devID].templateDepthFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*1*sizeof(unsigned short));
  }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibRGB);

  sprintf(file_name_test,"frames/%s/color.calib",device[devID].readFromDir);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibRGB) ) { fprintf(stderr,"Could not read color calibration\n"); }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibDepth);

  sprintf(file_name_test,"frames/%s/depth.calib",device[devID].readFromDir);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibDepth) ) { fprintf(stderr,"Could not read depth calibration\n"); }


  return ((device[devID].templateColorFrame!=0)&&(device[devID].templateDepthFrame!=0)&&(failedStream==0));
}



int destroyTemplateDevice(int devID)
{
  return 1;
}

int getTotalTemplateFrameNumber(int devID)
{
  return device[devID].totalFrames;
}

int getCurrentTemplateFrameNumber(int devID)
{
  return device[devID].cycle;
}


int snapTemplateFrames(int devID)
{
    #if PRINT_DEBUG_EACH_CALL
     fprintf(stderr,"snapTemplateFrames (%u) \n",devID);
    #endif // PRINT_DEBUG_EACH_CALL

    //TODO HERE MAYBE LOAD NEW BUFFERS
    int found_frames = 0;

    unsigned int widthInternal=0; unsigned int heightInternal=0;
    char * file_name_test = (char* ) malloc(2048 * sizeof(char));
    if (file_name_test==0) { fprintf(stderr,"Could not snap frame , no space for string\n"); return 0; }


    //-----------------------------------------------------------------
    //Extra check , stupid case with mixed signals
    //-----------------------------------------------------------------
    int decided=0;
    int devIDRead=devID;
    int devIDInc=devID;
    while ( (devIDInc >=0 ) && (!decided) )
    {
      sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devIDInc,device[devID].cycle);
      if (FileExists(file_name_test)) { devIDRead=devIDInc; decided=1; }
      sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devIDInc,device[devID].cycle);
      if (FileExists(file_name_test)) { devIDRead=devIDInc; decided=1; }

      if (devIDInc==0) { break; decided=1; } else
                       { --devIDInc; }
    }
    //-----------------------------------------------------------------


    sprintf(file_name_test,"frames/%s/colorFrame_%u_%05u.pnm",device[devID].readFromDir,devIDRead,device[devID].cycle);
    //fprintf(stderr,"Snap color %s\n",file_name_test);
    if (FileExists(file_name_test))
     {
       if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); }
       device[devID].templateColorFrame = ReadPPM(file_name_test,&widthInternal,&heightInternal,&device[devID].lastColorTimestamp);
       ++found_frames;
     }

    sprintf(file_name_test,"frames/%s/depthFrame_%u_%05u.pnm",device[devID].readFromDir,devIDRead,device[devID].cycle);
    //fprintf(stderr,"Snap depth %s",file_name_test);
    if (FileExists(file_name_test))
     {
      if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); }
      device[devID].templateDepthFrame = ReadPPMD(file_name_test,&widthInternal,&heightInternal,&device[devID].lastDepthTimestamp);
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



int seekRelativeTemplateFrame(int devID,signed int seekFrame)
{
  if (device[devID].cycle - seekFrame < 0 )  { device[devID].cycle=0; } else
                                             { device[devID].cycle += seekFrame; }
  return 1;
}

int seekTemplateFrame(int devID,unsigned int seekFrame)
{
  device[devID].cycle = seekFrame;
  return 1;
}


//Color Frame getters
unsigned long getLastTemplateColorTimestamp(int devID) { return device[devID].lastColorTimestamp; }
int getTemplateColorWidth(int devID)        { return device[devID].templateWIDTH; }
int getTemplateColorHeight(int devID)       { return device[devID].templateHEIGHT; }
int getTemplateColorDataSize(int devID)     { return device[devID].templateHEIGHT*device[devID].templateWIDTH * 3; }
int getTemplateColorChannels(int devID)     { return 3; }
int getTemplateColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getTemplateColorPixels(int devID)    { return device[devID].templateColorFrame; }




   //Depth Frame getters
unsigned long getLastTemplateDepthTimestamp(int devID) { return device[devID].lastDepthTimestamp; }
int getTemplateDepthWidth(int devID)    { return device[devID].templateWIDTH; }
int getTemplateDepthHeight(int devID)   { return device[devID].templateHEIGHT; }
int getTemplateDepthDataSize(int devID) { return device[devID].templateWIDTH*device[devID].templateHEIGHT; }
int getTemplateDepthChannels(int devID)     { return 1; }
int getTemplateDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getTemplateDepthPixels(int devID) { return (char *) device[devID].templateDepthFrame; }

char * getTemplateDepthPixelsFlipped(int devID) {
                                                  flipDepth(device[devID].templateDepthFrame,device[devID].templateWIDTH, device[devID].templateHEIGHT);
                                                  return (char *) device[devID].templateDepthFrame;
                                                }

#else
//Null build
int startTemplateModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startTemplateModule called on a dummy build of TemplateAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_TEMPLATE 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
