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

#include "templateAcquisitionHelper.h"

#define SAFEGUARD_VALUE 123123
#define MAX_TEMPLATE_DEVICES 5
#define MAX_LINE_CALIBRATION 1024

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

#define REALLOCATE_ON_EVERY_SNAP 0
 #if REALLOCATE_ON_EVERY_SNAP
         #warning "Reallocating arrays on every frame ,  performance will be impacted by this "
        #else
         #warning "Not Reallocating arrays on every frame this means better performance but if stream dynamically changes resolution from frame to frame this could be a problem"
 #endif




struct TemplateVirtualDevice
{
 char readFromDir[MAX_DIR_PATH]; // <- this sucks i know :P
 char extension[MAX_EXTENSION_PATH];
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



int mapTemplateDepthToRGB(int devID) { return 0; }
int mapTemplateRGBToDepth(int devID) { return 0; }
int switchTemplateToColorStream(int devID) { return 1; }

int getTemplateNumberOfColorStreams(int devID) { return 1; /*TODO support multiple streams*/ }

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





int startTemplateModule(unsigned int max_devs,char * settings)
{
    unsigned int devID = 0;
    for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
    {
        //fprintf(stderr,"Zeroing device %u\n",devID);
        device[devID].templateWIDTH = 640;
        device[devID].templateHEIGHT = 480;

        device[devID].readFromDir[0]=0; // <- this sucks i know :P
        strncpy(device[devID].extension,"pnm",MAX_EXTENSION_PATH);

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

  findExtensionOfDataset(devID,device[devID].readFromDir,device[devID].extension);
  fprintf(stderr,"Extension of dataset %s is %s \n",device[devID].readFromDir,device[devID].extension);
  device[devID].totalFrames=findLastFrame(devID,device[devID].readFromDir,device[devID].extension);

  unsigned int failedStream=0;
  unsigned int widthInternal=0; unsigned int heightInternal=0; unsigned long timestampInternal=0;

  char file_name_test[MAX_DIR_PATH];
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devID , 0 ,device[devID].readFromDir,device[devID].extension);
  unsigned char * tmpColor = ReadImageFile(0,file_name_test,device[devID].extension,&widthInternal,&heightInternal, &timestampInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
       { fprintf(stderr,"Please note that the %s file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",file_name_test,widthInternal,heightInternal,width,height); }

  if (tmpColor!=0) { device[devID].templateColorFrame=tmpColor; } else
  {
   ++failedStream;
   // if templateColorFrame is zero the next function behaves like a malloc
   device[devID].templateColorFrame= (unsigned char*) realloc(device[devID].templateColorFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*3*sizeof(char));
   makeFrameNoInput(device[devID].templateColorFrame,device[devID].templateWIDTH,device[devID].templateHEIGHT,3);
  }


  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devID ,0,device[devID].readFromDir,device[devID].extension);
  unsigned short * tmpDepth = (unsigned short *) ReadImageFile(0,file_name_test,device[devID].extension,&widthInternal,&heightInternal, &timestampInternal);
  if ( (widthInternal!=width) || (heightInternal!=height) )
   { fprintf(stderr,"Please note that the %s file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n",file_name_test,widthInternal,heightInternal,width,height); }

  if (tmpDepth!=0) { device[devID].templateDepthFrame=tmpDepth; } else
  {
   ++failedStream;
   // if templateDepthFrame is zero the next function behaves like a malloc
   device[devID].templateDepthFrame= (unsigned short*) realloc(device[devID].templateDepthFrame,device[devID].templateWIDTH*device[devID].templateHEIGHT*1*sizeof(unsigned short));
  }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibRGB);
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_CALIBRATION_FILE , devID ,device[devID].cycle,device[devID].readFromDir,device[devID].extension);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibRGB) ) { fprintf(stderr,"Could not read color calibration\n"); }

  NullCalibration(device[devID].templateWIDTH,device[devID].templateHEIGHT,&device[devID].calibDepth);
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_CALIBRATION_FILE , devID ,device[devID].cycle,device[devID].readFromDir,device[devID].extension);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibDepth) ) { fprintf(stderr,"Could not read depth calibration\n"); }

  if (device[devID].templateColorFrame==0) { fprintf(stderr,RED " Could not open , color frame Template acquisition will not process this data\n"); }
  if (device[devID].templateDepthFrame==0) { fprintf(stderr,RED " Could not open , depth frame Template acquisition will not process this data\n"); }
  return ((device[devID].templateColorFrame!=0)&& (device[devID].templateDepthFrame!=0)&& (failedStream==0));
}



int destroyTemplateDevice(int devID)
{
  if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); device[devID].templateColorFrame=0; }
  if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); device[devID].templateDepthFrame=0; }
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
    //TODO HERE MAYBE LOAD NEW BUFFERS
    int found_frames = 0;

    //-----------------------------------------------------------------
    //Extra check , stupid case with mixed signals
    //-----------------------------------------------------------------
    unsigned int devIDRead = retreiveDatasetDeviceIDToReadFrom( devID , device[devID].cycle , device[devID].readFromDir , device[devID].extension);
    //-----------------------------------------------------------------


    unsigned int widthInternal=0; unsigned int heightInternal=0;
    char * file_name_test = (char* ) malloc(MAX_DIR_PATH * sizeof(char));
    if (file_name_test==0) { fprintf(stderr,"Could not snap frame , no space for string\n"); return 0; }

    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_LIVE_CALIBRATION_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,device[devID].extension);
    if ( RefreshCalibration(file_name_test,&device[devID].calibRGB) )
     {
       fprintf(stderr,"Refreshed calibration data %u \n",device[devID].cycle);
     }

    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,device[devID].extension);
    if (FileExists(file_name_test))
     {
       #if REALLOCATE_ON_EVERY_SNAP
         if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); device[devID].templateColorFrame=0; }
       #endif
       device[devID].templateColorFrame = ReadImageFile(device[devID].templateColorFrame,file_name_test,device[devID].extension,&widthInternal,&heightInternal,&device[devID].lastColorTimestamp);
       ++found_frames;
     }

    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,device[devID].extension);
    if (FileExists(file_name_test))
     {
      #if REALLOCATE_ON_EVERY_SNAP
        if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); device[devID].templateDepthFrame=0; }
      #endif
      device[devID].templateDepthFrame = (unsigned short *) ReadImageFile(device[devID].templateDepthFrame,file_name_test,device[devID].extension,&widthInternal,&heightInternal,&device[devID].lastColorTimestamp);
      ++found_frames;
     }

  free(file_name_test);
  file_name_test=0;

  ++device[devID].cycle;
  if ( device[devID].safeGUARD != SAFEGUARD_VALUE ) { fprintf(stderr,"\n\n\n\nERROR , memory corruption \n\n\n\n"); }

  if (device[devID].cycle>65534) { device[devID].cycle=0; }
  if (found_frames==0) { fprintf(stderr,YELLOW "Could not find any frames , we finished stream \n" NORMAL);  device[devID].cycle = 0; } else
  if (found_frames!=2) { fprintf(stderr,YELLOW "\n Warning: Did not find both frames\n" NORMAL);   }

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
