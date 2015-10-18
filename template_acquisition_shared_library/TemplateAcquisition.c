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
 char intialized;
 char readFromDir[MAX_DIR_PATH]; // <- this sucks i know :P
 char colorExtension[MAX_EXTENSION_PATH];
 char depthExtension[MAX_EXTENSION_PATH];
 unsigned int cycle;
 float cycleFlow;
 unsigned int totalFrames;
 unsigned int safeGUARD;


 unsigned int templateColorWidth;
 unsigned int templateColorHeight;

 unsigned int templateDepthWidth;
 unsigned int templateDepthHeight;

 unsigned long lastColorTimestamp;
 unsigned char * templateColorFrame;
 unsigned long lastDepthTimestamp;
 unsigned short * templateDepthFrame;

 struct calibration calibRGB;
 unsigned int disableRGBStream;
 struct calibration calibDepth;
 unsigned int disableDepthStream;

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



int enableTemplateStream(int devID,unsigned int streamID)
{
    if (streamID==0) { device[devID].disableRGBStream=0; } else
    if (streamID==1) { device[devID].disableDepthStream=0; }
    return 1;
}

int disableTemplateStream(int devID,unsigned int streamID)
{
    if (streamID==0) { device[devID].disableRGBStream=1; } else
    if (streamID==1) { device[devID].disableDepthStream=1; }
    return 1;
}



int startTemplateModule(unsigned int max_devs,char * settings)
{
    unsigned int devID = 0;
    for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
    {
        //fprintf(stderr,"Zeroing device %u\n",devID);
        device[devID].templateColorWidth = 640;
        device[devID].templateColorHeight = 480;

        device[devID].templateDepthWidth = 640;
        device[devID].templateDepthHeight = 480;

        device[devID].readFromDir[0]=0; // <- this sucks i know :P
        strncpy(device[devID].colorExtension,"pnm",MAX_EXTENSION_PATH);
        strncpy(device[devID].depthExtension,"pnm",MAX_EXTENSION_PATH);

        device[devID].cycle=0;
        device[devID].intialized=0;

        device[devID].safeGUARD = SAFEGUARD_VALUE;

        device[devID].templateColorFrame=0;
        device[devID].templateDepthFrame=0;
    }

    fprintf(stderr,"startTemplate done \n");
    return 1;
}

int deviceIsSafeToUse(int devID)
{
  if ( devID<MAX_TEMPLATE_DEVICES )
  {
    return 1;
    #warning "For now only do a bounds check"
    if (device[devID].intialized)
      {
        return 1;
      }
  }
 return 0;
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
 // We may want to start from a non zero frame : device[devID].cycle=0;
 fprintf(stderr,"Creating a template device starting from frame %u \n",device[devID].cycle);
 device[devID].totalFrames=0;
 device[devID].templateColorWidth=width;
 device[devID].templateColorHeight=height;
 device[devID].templateDepthWidth=width;
 device[devID].templateDepthHeight=height;
 device[devID].colorExtension[0]=0;
 device[devID].depthExtension[0]=0;


   if (devName==0) { strcpy(device[devID].readFromDir,""); } else
     {
       if (strlen(devName)==0)  { strcpy(device[devID].readFromDir,""); } else
                                { strncpy(device[devID].readFromDir,devName,MAX_DIR_PATH);  }
     }

  findExtensionOfDataset(devID,device[devID].readFromDir,device[devID].colorExtension,device[devID].depthExtension,device[devID].cycle);
  fprintf(stderr,"Extension of dataset `%s` dev %u for Color Frames is %s , starting @ %u\n",device[devID].readFromDir,devID,device[devID].colorExtension,device[devID].cycle);
  fprintf(stderr,"Extension of dataset `%s` dev %u for Depth Frames is %s , starting @ %u\n",device[devID].readFromDir,devID,device[devID].depthExtension,device[devID].cycle);


  device[devID].totalFrames=findLastFrame(devID,device[devID].readFromDir,device[devID].colorExtension,device[devID].depthExtension);
  fprintf(stderr,"Dataset %s consists of %u frame pairs \n",device[devID].readFromDir,device[devID].totalFrames);

  unsigned int failedStream=0;
  unsigned int widthInternal=0; unsigned int heightInternal=0; unsigned long timestampInternal=0;

  char file_name_test[MAX_DIR_PATH]={0};
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devID , device[devID].cycle ,device[devID].readFromDir,device[devID].colorExtension);
  unsigned char * tmpColor = ReadImageFile(0,file_name_test,device[devID].colorExtension,&widthInternal,&heightInternal, &timestampInternal);
  if (tmpColor==0) { fprintf(stderr,YELLOW "Could not open initial color file %s \n",file_name_test);  }
  if ( (widthInternal!=width) || (heightInternal!=height) )
    {
      fprintf(stderr,YELLOW "Please note that the %s file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n" NORMAL,file_name_test,widthInternal,heightInternal,width,height);
      device[devID].templateColorWidth=widthInternal;
      device[devID].templateColorHeight=heightInternal;
    }


  if (tmpColor!=0) { device[devID].templateColorFrame=tmpColor; } else
  {
   ++failedStream;
   // if templateColorFrame is zero the next function behaves like a malloc
   device[devID].templateColorFrame= (unsigned char*) realloc(device[devID].templateColorFrame,device[devID].templateColorWidth*device[devID].templateColorHeight*3*sizeof(char));
   makeFrameNoInput(device[devID].templateColorFrame,device[devID].templateColorWidth,device[devID].templateColorHeight,3);
  }


  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devID ,device[devID].cycle,device[devID].readFromDir,device[devID].depthExtension);
  unsigned short * tmpDepth = (unsigned short *) ReadImageFile(0,file_name_test,device[devID].depthExtension,&widthInternal,&heightInternal, &timestampInternal);
  if (tmpDepth==0) { fprintf(stderr,YELLOW "Could not open initial depth file %s \n",file_name_test);  }
  if ( (widthInternal!=width) || (heightInternal!=height) )
   {
    fprintf(stderr,YELLOW "Please note that the %s file has %ux%u resolution and the createTemplateDevice asked for %ux%u \n" NORMAL,file_name_test,widthInternal,heightInternal,width,height);
      device[devID].templateDepthWidth=widthInternal;
      device[devID].templateDepthHeight=heightInternal;
   }



  if (tmpDepth!=0) { device[devID].templateDepthFrame=tmpDepth; } else
  {
   ++failedStream;
   // if templateDepthFrame is zero the next function behaves like a malloc
   device[devID].templateDepthFrame= (unsigned short*) realloc(device[devID].templateDepthFrame,device[devID].templateDepthWidth*device[devID].templateDepthHeight*1*sizeof(unsigned short));
  }

  NullCalibration(device[devID].templateColorWidth,device[devID].templateColorHeight,&device[devID].calibRGB);
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_CALIBRATION_FILE , devID ,device[devID].cycle,device[devID].readFromDir,device[devID].colorExtension);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibRGB) ) { fprintf(stderr,"Could not read color calibration\n"); }

  NullCalibration(device[devID].templateDepthWidth,device[devID].templateDepthHeight,&device[devID].calibDepth);
  getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_CALIBRATION_FILE , devID ,device[devID].cycle,device[devID].readFromDir,device[devID].depthExtension);
  if ( ! ReadCalibration(file_name_test,widthInternal,heightInternal,&device[devID].calibDepth) ) { fprintf(stderr,"Could not read depth calibration\n"); }

  if (device[devID].templateColorFrame==0) { fprintf(stderr,RED " Could not open , color frame Template acquisition will not process this data\n"); }
  if (device[devID].templateDepthFrame==0) { fprintf(stderr,RED " Could not open , depth frame Template acquisition will not process this data\n"); }

  if (failedStream)
  {
   #ifdef USE_CODEC_LIBRARY
    fprintf(stderr,YELLOW "We were using Codec Library\n" NORMAL);
   #else
    fprintf(stderr,YELLOW "Please note that this build of TemplateAcquisition does not use the codec library and thus cannot read images other than in pnm format\n" NORMAL);
    fprintf(stderr,YELLOW "Formats used by this dataset where %s for color and %s for depth\n" NORMAL,device[devID].colorExtension,device[devID].depthExtension);
   #endif // USE_CODEC_LIBRARY
  }

  if ( (device[devID].disableRGBStream) )  { fprintf(stderr,GREEN "RGB Stream is disabled so we will take that into account \n" NORMAL); }
  if ( (device[devID].disableDepthStream) )  { fprintf(stderr,GREEN "Depth Stream is disabled so we will take that into account \n" NORMAL); }

  device[devID].intialized = (
                               ( (device[devID].templateColorFrame!=0) || (device[devID].disableRGBStream) )&&
                               ( (device[devID].templateDepthFrame!=0) || (device[devID].disableDepthStream) ) &&
                               ( (failedStream==0) || (device[devID].disableRGBStream) || (device[devID].disableDepthStream) )
                             );

   if (!device[devID].intialized)
   {
     fprintf(stderr,"Consider running with -noColor or -noDepth if a missing stream is the problem \n");
   }

  return device[devID].intialized;
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
    unsigned int devIDRead = retreiveDatasetDeviceIDToReadFrom( devID , device[devID].cycle , device[devID].readFromDir , device[devID].colorExtension);
    //-----------------------------------------------------------------


    unsigned int widthInternal=0; unsigned int heightInternal=0;
    char * file_name_test = (char* ) malloc(MAX_DIR_PATH * sizeof(char));
    if (file_name_test==0) { fprintf(stderr,"Could not snap frame , no space for string\n"); return 0; }


    //TODO : Check the next line , does it make sense ?
    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_LIVE_CALIBRATION_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,0/*calib files are special*/);
    if ( RefreshCalibration(file_name_test,&device[devID].calibRGB) )
     {
       fprintf(stderr,"Refreshed calibration data %u \n",device[devID].cycle);
     }

    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_COLOR_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,device[devID].colorExtension);
    if (FileExists(file_name_test))
     {
       #if REALLOCATE_ON_EVERY_SNAP
         if (device[devID].templateColorFrame!=0) { free(device[devID].templateColorFrame); device[devID].templateColorFrame=0; }
       #endif
       device[devID].templateColorFrame = ReadImageFile(device[devID].templateColorFrame,file_name_test,device[devID].colorExtension,&widthInternal,&heightInternal,&device[devID].lastColorTimestamp);
       ++found_frames;
     }

    getFilenameForNextResource(file_name_test , MAX_DIR_PATH , RESOURCE_DEPTH_FILE , devIDRead ,device[devID].cycle,device[devID].readFromDir,device[devID].depthExtension);
    if (FileExists(file_name_test))
     {
      #if REALLOCATE_ON_EVERY_SNAP
        if (device[devID].templateDepthFrame!=0) { free(device[devID].templateDepthFrame); device[devID].templateDepthFrame=0; }
      #endif
      device[devID].templateDepthFrame = (unsigned short *) ReadImageFile(device[devID].templateDepthFrame,file_name_test,device[devID].depthExtension,&widthInternal,&heightInternal,&device[devID].lastColorTimestamp);
      ++found_frames;
     }

  free(file_name_test);
  file_name_test=0;


  if (device[devID].cycleFlow<0.0)
  {
   if (device[devID].cycle>0) { --device[devID].cycle; }
  } else
  {
   ++device[devID].cycle;
   if (device[devID].cycle>65534) { device[devID].cycle=0; }
  }


  if ( device[devID].safeGUARD != SAFEGUARD_VALUE ) { fprintf(stderr,"\n\n\n\nERROR , memory corruption \n\n\n\n"); }

  if (found_frames==0) { /*fprintf(stderr,YELLOW "Finished stream \n" NORMAL);*/  device[devID].cycle = 0; } else
  if (found_frames!=2) { fprintf(stderr,YELLOW "\n Warning: Did not find both frames\n" NORMAL);   }

  return 1;
}


int controlTemplateFlow(int devID,float newFlowState)
{
  device[devID].cycleFlow = newFlowState;
}


int seekRelativeTemplateFrame(int devID,signed int seekFrame)
{
  if (!deviceIsSafeToUse(devID)) { fprintf(stderr,YELLOW "Device %u is not safe to use At %s , Line %u" NORMAL , __FILE__ , __LINE__ ); return 0; }

  if (device[devID].cycle + seekFrame < 0 )  { device[devID].cycle=0; } else
                                             { device[devID].cycle += seekFrame; }
  return 1;
}

int seekTemplateFrame(int devID,unsigned int seekFrame)
{
  if (!deviceIsSafeToUse(devID)) { fprintf(stderr,YELLOW "Device %u is not safe to use At %s , Line %u" NORMAL , __FILE__ , __LINE__ ); return 0; }

  device[devID].cycle = seekFrame;
  return 1;
}


//Color Frame getters
unsigned long getLastTemplateColorTimestamp(int devID) { return device[devID].lastColorTimestamp; }
int getTemplateColorWidth(int devID)        { return device[devID].templateColorWidth; }
int getTemplateColorHeight(int devID)       { return device[devID].templateColorHeight; }
int getTemplateColorDataSize(int devID)     { return device[devID].templateColorHeight*device[devID].templateColorWidth * 3; }
int getTemplateColorChannels(int devID)     { return 3; }
int getTemplateColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getTemplateColorPixels(int devID)    { return device[devID].templateColorFrame; }




   //Depth Frame getters
unsigned long getLastTemplateDepthTimestamp(int devID) { return device[devID].lastDepthTimestamp; }
int getTemplateDepthWidth(int devID)    { return device[devID].templateDepthWidth; }
int getTemplateDepthHeight(int devID)   { return device[devID].templateDepthHeight; }
int getTemplateDepthDataSize(int devID) { return device[devID].templateDepthWidth*device[devID].templateDepthHeight; }
int getTemplateDepthChannels(int devID)     { return 1; }
int getTemplateDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getTemplateDepthPixels(int devID) { return (char *) device[devID].templateDepthFrame; }

char * getTemplateDepthPixelsFlipped(int devID) {
                                                  flipDepth(device[devID].templateDepthFrame,device[devID].templateDepthWidth, device[devID].templateDepthHeight);
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
