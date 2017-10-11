#include <stdio.h>
#include <stdlib.h>

#include "OpenNI1Acquisition.h"
#define BUILD_OPENNI1 1

#if BUILD_OPENNI1

#include <XnOS.h>
#include <XnCppWrapper.h>
#include <XnLog.h>

/* This does not work ..
if you get an error that you can't find XnOS.h just make sure the makefile is correct..!
#include <ni/XnOS.h>
#include <ni/XnCppWrapper.h>
#include <ni/XnLog.h>
*/

using namespace std;
using namespace xn;


//#define SAMPLE_XML_PATH "OpenNIConfig.xml"
#define MAX_OPENNI_DEVICES 16
#define ANY_OPENNI_DEVICE MAX_OPENNI_DEVICES*2

Device devices[MAX_OPENNI_DEVICES];
DepthGenerator depthGenerators[MAX_OPENNI_DEVICES]={0};
ImageGenerator imageGenerators[MAX_OPENNI_DEVICES]={0};

DepthMetaData depthGeneratorsMetaData[MAX_OPENNI_DEVICES];
ImageMetaData imageGeneratorsMetaData[MAX_OPENNI_DEVICES];
ScriptNode script;
Context ctx;


#if USE_CALIBRATION
 struct calibration calibRGB[MAX_OPENNI_DEVICES];
 struct calibration calibDepth[MAX_OPENNI_DEVICES];
#endif


int startOpenNI1Module(unsigned int max_devs,char * settings)
{

 EnumerationErrors errors;
 XnStatus rc;

   unsigned int useXMLFile=0;
   if (settings!=0)
   {
    if (strstr(settings,".ini")!=0) { useXMLFile=1; } else
    if (strstr(settings,".xml")!=0) { useXMLFile=1; }
   }

   if (useXMLFile) { rc = ctx.InitFromXmlFile(settings); } else
                   { rc = ctx.Init(); }

 if (rc == XN_STATUS_NO_NODE_PRESENT)
  {
    XnChar strError[1024];
    errors.ToString(strError, 1024);
    printf("No devices present : %s\n", strError);
    return 0;
   }
    else
 if (rc != XN_STATUS_OK)
  {
   printf("Open failed: %s\n", xnGetStatusString(rc));
   return 0;
  }


 /*
  // find devices
  NodeInfoList list;
  XnStatus nRetVal = XN_STATUS_OK;
  nRetVal = ctx.EnumerateProductionTrees(XN_NODE_TYPE_DEVICE, NULL, list, &errors);
  XN_IS_STATUS_OK(nRetVal);

  printf("The following devices were found:\n");
  int i = 1;
  for (NodeInfoList::Iterator it = list.Begin(); it != list.End(); ++it, ++i)
   {
    NodeInfo deviceNodeInfo = *it;

    Device deviceNode;
    deviceNodeInfo.GetInstance(deviceNode);
    XnBool bExists = deviceNode.IsValid();
    if (!bExists)
     {
       ctx.CreateProductionTree(deviceNodeInfo, deviceNode); // this might fail.
     }

    if (deviceNode.IsValid() && deviceNode.IsCapabilitySupported(XN_CAPABILITY_DEVICE_IDENTIFICATION))
     {
      const XnUInt32 nStringBufferSize = 200;
      XnChar strDeviceName[nStringBufferSize];
      XnChar strSerialNumber[nStringBufferSize];

      XnUInt32 nLength = nStringBufferSize;
      deviceNode.GetIdentificationCap().GetDeviceName(strDeviceName, nLength);
      nLength = nStringBufferSize;
      deviceNode.GetIdentificationCap().GetSerialNumber(strSerialNumber, nLength);
      printf("[%d] %s (%s)\n", i, strDeviceName, strSerialNumber);
    }
     else
   {
    printf("[%d] %s\n", i, deviceNodeInfo.GetCreationInfo());
   }

   // release the device if we created it

   if (!bExists && deviceNode.IsValid())
    {
     deviceNode.Release();
    }

 }

 */



 return 1;
}





int mapOpenNI1DepthToRGB(int devID)
{
  printf("mapOpenNI1DepthToRGB\n");
  if (!depthGenerators[devID]) { return 0; }
  if (!imageGenerators[devID]) { return 0; }

  XnBool isSupported = depthGenerators[devID].IsCapabilitySupported("AlternativeViewPoint");
  if(isSupported)
   {
      XnStatus res = depthGenerators[devID].GetAlternativeViewPointCap().SetViewPoint(imageGenerators[devID]);
      if(XN_STATUS_OK != res)
      {
         printf("Getting and setting AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
         return 0;
      }
     return 1;
   }
 return 0;
}


int mapOpenNI1RGBToDepth(int devID)
{
  printf("mapOpenNI1RGBToDepth\n");
  if (!imageGenerators[devID]) { return 0; }
  if (!depthGenerators[devID]) { return 0; }

  XnBool isSupported = imageGenerators[devID].IsCapabilitySupported("AlternativeViewPoint");
  if(isSupported)
   {
      XnStatus res = imageGenerators[devID].GetAlternativeViewPointCap().SetViewPoint(depthGenerators[devID]);
      if(XN_STATUS_OK != res)
      {
         printf("Getting and setting AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
         return 0;
      }
     return 1;
   }
 return 0;
}


int getOpenNI1NumberOfDevices()  {  fprintf(stderr,"getOpenNI1NumberOfDevices is a stub it always returns 1\n");  return 1; }


int stopOpenNI1Module()
{
  ctx.Release();

  return 1;
}


int SignalOpenNIError(char * description , XnStatus rc)
{
  if (rc != XN_STATUS_OK) { printf("Error : %s ( %s )\n",description,xnGetStatusString(rc)); return 1; }
  return 0;
}


   //Basic Per Device Operations
int createOpenNI1Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
    XnStatus rc;
    XnMapOutputMode mapMode;

/*------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------ */
    rc = depthGenerators[devID].Create(ctx);
    if ( SignalOpenNIError("Could not create a new depth generator",rc) ) { return 0; }
   // rc = ctx.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerators[devID]);
   // if ( SignalOpenNIError("No depth node exists!",rc) ) { return 0; }


    mapMode.nXRes = width; mapMode.nYRes = height; mapMode.nFPS = framerate;
    rc = depthGenerators[devID].SetMapOutputMode(mapMode);
    SignalOpenNIError("Could not set output mode for depth ",rc);

    rc = depthGenerators[devID].StartGenerating();
    if (rc != XN_STATUS_OK) {  SignalOpenNIError("Could not start generating depth output",rc);  }

    depthGenerators[devID].GetMirrorCap().SetMirror(false);
    depthGenerators[devID].GetMetaData(depthGeneratorsMetaData[devID]);

    fprintf(stderr,"Depth grabber @ %ux%u\n",depthGeneratorsMetaData[devID].FullXRes(),depthGeneratorsMetaData[devID].FullYRes());


/*------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------ */
    rc = imageGenerators[devID].Create(ctx);
    if ( SignalOpenNIError("Could not create a new image generator",rc) ) { return 0; }
  //  rc = ctx.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerators[devID]);
  //  if ( SignalOpenNIError("No image node exists!",rc) ) { return 0; }

    mapMode.nXRes = width; mapMode.nYRes = height; mapMode.nFPS = framerate;
    rc = imageGenerators[devID].SetMapOutputMode(mapMode);
    SignalOpenNIError("Could not set output mode for image ",rc);

    if ( imageGenerators[devID].IsPixelFormatSupported(XN_PIXEL_FORMAT_RGB24) )
    {
      rc = imageGenerators[devID].SetPixelFormat(XN_PIXEL_FORMAT_RGB24);
      if (rc != XN_STATUS_OK) {  SignalOpenNIError("Could not set format to RGB24 ",rc);  }
    } else
    { fprintf(stderr,"Device does not Support RGB24 output \n"); }

    rc = imageGenerators[devID].StartGenerating();
    if (rc != XN_STATUS_OK) {  SignalOpenNIError("Could not start generating image output",rc);  }


    imageGenerators[devID].GetMirrorCap().SetMirror(false);
    imageGenerators[devID].GetMetaData(imageGeneratorsMetaData[devID]);


    fprintf(stderr,"Image grabber @ %ux%u\n",imageGeneratorsMetaData[devID].FullXRes(),imageGeneratorsMetaData[devID].FullYRes());


    //rc = ctx.StartGeneratingAll();
    //if (rc != XN_STATUS_OK) {  SignalOpenNIError("Could not start generating everything",rc);  }


    // Hybrid mode isn't supported in this sample
if ( ( imageGeneratorsMetaData[devID].FullXRes() != depthGeneratorsMetaData[devID].FullXRes()) ||
      (imageGeneratorsMetaData[devID].FullYRes() != depthGeneratorsMetaData[devID].FullYRes())  )
   { printf ("The device depth and image resolution are not equal!\n"); }


// RGB is the only image format supported.
if (imageGeneratorsMetaData[devID].PixelFormat() != XN_PIXEL_FORMAT_RGB24)
  {
    printf("The device image format is not RGB24\n");
    return 0;
  }

    #if USE_CALIBRATION
     printf("Populating calibration data\n");
     //Populate our calibration data ( if we have them
     FocalLengthAndPixelSizeToCalibration(getOpenNI1ColorFocalLength(devID),getOpenNI1ColorFocalLength(devID),getOpenNI1ColorWidth(devID),getOpenNI1ColorHeight(devID),&calibRGB[devID]);
     FocalLengthAndPixelSizeToCalibration(getOpenNI1DepthFocalLength(devID),getOpenNI1DepthPixelSize(devID),getOpenNI1DepthWidth(devID),getOpenNI1DepthHeight(devID),&calibDepth[devID]);
    #endif



 fprintf(stderr,"We seem to have correctly initialized OpenNI1 %u , %s \n",devID,devName);
 return 1;
}

int destroyOpenNI1Device(int devID) { return 0; }

int snapOpenNI1Frames(int devID)
{
  XnStatus rc = XN_STATUS_OK;
 // Read a new frame
 //rc = ctx.WaitAnyUpdateAll();
 //if (rc != XN_STATUS_OK) { printf("Read failed: %s\n", xnGetStatusString(rc)); return 0; }


    if (imageGenerators[devID])
      {
       rc = imageGenerators[devID].WaitAndUpdateData();
       if (rc != XN_STATUS_OK) { printf("Image Generator could not wait for new data ( %s )\n",xnGetStatusString(rc)); return 0; }
       imageGenerators[devID].GetMetaData(imageGeneratorsMetaData[devID]);
      }

    if (depthGenerators[devID])
      {
       rc = depthGenerators[devID].WaitAndUpdateData();
       if (rc != XN_STATUS_OK) { printf("Depth Generator could not wait for new data ( %s )\n",xnGetStatusString(rc)); return 0; }
       depthGenerators[devID].GetMetaData(depthGeneratorsMetaData[devID]);
      }

 return 1;
}

//Color Frame getters
int getOpenNI1ColorWidth(int devID) { return imageGeneratorsMetaData[devID].FullXRes(); }
int getOpenNI1ColorHeight(int devID) { return imageGeneratorsMetaData[devID].FullYRes(); }
int getOpenNI1ColorDataSize(int devID) { return getOpenNI1ColorWidth(devID)*getOpenNI1ColorHeight(devID)*3; }
int getOpenNI1ColorChannels(int devID) { return 3; }
int getOpenNI1ColorBitsPerPixel(int devID) { return 8; }
char * getOpenNI1ColorPixels(int devID)
{
    //return (char*) imageGenerators[devID].GetImageMap();
    //return (char*) imageGeneratorsMetaData[devID].Data();
    //return (char*) imageGenerators[devID].GetRGB24ImageMap();
    return (char*) imageGeneratorsMetaData[devID].RGB24Data();
}


double getOpenNI1ColorFocalLength(int devID)
{
    XnUInt64 focalLength;
	// get the focal length in mm (ZPS = zero plane distance)
	imageGenerators[devID].GetIntProperty ("ZPD", focalLength);
    return (double) focalLength;
}

double getOpenNI1ColorPixelSize(int devID)
{
	XnDouble pixelSize;
	// get the pixel size in mm ("ZPPS" = pixel size at zero plane)
	imageGenerators[devID].GetRealProperty ("ZPPS", pixelSize);
	fprintf(stderr,"Note : OpenNI1 gives us half the true pixel size ? ? \n");
    pixelSize*=2.0;
    return (double) pixelSize;
}



//Depth Frame getters
int getOpenNI1DepthWidth(int devID) { return depthGeneratorsMetaData[devID].FullXRes(); }
int getOpenNI1DepthHeight(int devID) { return depthGeneratorsMetaData[devID].FullYRes(); }
int getOpenNI1DepthDataSize(int devID) { return getOpenNI1DepthWidth(devID)*getOpenNI1DepthHeight(devID); }
int getOpenNI1DepthChannels(int devID) { return 1; }
int getOpenNI1DepthBitsPerPixel(int devID) { return 16; }
short * getOpenNI1DepthPixels(int devID)
{
  return (short*) depthGenerators[devID].GetDepthMap();
  //return (short*) depthGeneratorsMetaData[devID].Data();
  //return (short*) depthGenerators[devID].GetDepthMap();
  //return (short*) depthGeneratorsMetaData[devID].Data();
}

double getOpenNI1DepthFocalLength(int devID)
{
    XnUInt64 focalLength;
	// get the focal length in mm (ZPS = zero plane distance)
	depthGenerators[devID].GetIntProperty ("ZPD", focalLength);
    return (double) focalLength;
}

double getOpenNI1DepthPixelSize(int devID)
{
	XnDouble pixelSize;
	// get the pixel size in mm ("ZPPS" = pixel size at zero plane)
	depthGenerators[devID].GetRealProperty ("ZPPS", pixelSize);
	fprintf(stderr,"Note : OpenNI1 gives us half the true pixel size ? ? \n");
    pixelSize*=2.0;
    return (double) pixelSize;
}




#if USE_CALIBRATION
int getOpenNI1ColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibRGB[devID],sizeof(struct calibration));
    return 1;
}

int getOpenNI1DepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibDepth[devID],sizeof(struct calibration));
    return 1;
}


int setOpenNI1ColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &calibRGB[devID] , (void*) calib,sizeof(struct calibration));
    return 1;
}

int setOpenNI1DepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &calibDepth[devID] , (void*) calib,sizeof(struct calibration));
    return 1;
}
#endif


#else
//Null build
int startOpenNI1Module(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startOpenNI1Module called on a dummy build of OpenNI1Acquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_OPENNI1 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
