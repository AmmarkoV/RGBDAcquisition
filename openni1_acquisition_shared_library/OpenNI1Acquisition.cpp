#include <stdio.h>
#include <stdlib.h>

#include <XnOS.h>
#include <XnCppWrapper.h>
#include <XnLog.h>

#include "OpenNI1Acquisition.h"

using namespace std;
using namespace xn;


#define SAMPLE_XML_PATH "OpenNIConfig.xml"
#define MAX_OPENNI_DEVICES 16
#define ANY_OPENNI_DEVICE MAX_OPENNI_DEVICES*2

Device devices[MAX_OPENNI_DEVICES];
DepthGenerator depthGenerators[MAX_OPENNI_DEVICES];
ImageGenerator imageGenerators[MAX_OPENNI_DEVICES];

DepthMetaData depthGeneratorsMetaData[MAX_OPENNI_DEVICES];
ImageMetaData imageGeneratorsMetaData[MAX_OPENNI_DEVICES];
ScriptNode script;
Context ctx;


int mapOpenNI1DepthToRGB(int devID)
{
  if (!depthGenerators[devID]) { return 0; }
  depthGenerators[devID].GetAlternativeViewPointCap().SetViewPoint(imageGenerators[devID]);
  return 1;
}


int mapOpenNI1RGBToDepth(int devID)
{
  imageGenerators[devID].GetAlternativeViewPointCap().SetViewPoint(depthGenerators[devID]);
  return 1;
}

int startOpenNI1(unsigned int max_devs)
{
EnumerationErrors errors;
XnStatus rc;
//rc = ctx.InitFromXmlFile("SamplesConfig.xml");
rc = ctx.Init();

if (rc == XN_STATUS_NO_NODE_PRESENT)
{
   XnChar strError[1024];
   errors.ToString(strError, 1024);
   printf("%s\n", strError);
   return 0;
  }
else
if (rc != XN_STATUS_OK)
 {
   printf("Open failed: %s\n", xnGetStatusString(rc));
   return 0;
  }


/*
    xn::HandsGenerator handsGen;
    unsigned int nRetVal = ctx.CreateAnyProductionTree(XN_NODE_TYPE_HANDS, NULL, handsGen, &errors);
    if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
        {
           // Iterate over enumeration errors, and print each one
           for (xn::EnumerationErrors::Iterator it = errors.Begin(); it != errors.End(); ++it)
             {
                XnChar strDesc[512];
                xnProductionNodeDescriptionToString(&it.Description(), strDesc,512);
                printf("%s failed to enumerate: %s\n",xnGetStatusString(it.Error()));
             }
           return (nRetVal);
        } else
    if (nRetVal != XN_STATUS_OK)
        {
           printf("Create failed: %s\n", xnGetStatusString(nRetVal));
           return (nRetVal);
        }
*/

 return 1;
}



int getOpenNI1NumberOfDevices()  {  fprintf(stderr,"getOpenNI1NumberOfDevices is a stub it always returns 1");  return 1; }


int stopOpenNI1()
{
  ctx.Release();

  return 1;
}

   //Basic Per Device Operations
int createOpenNI1Device(int devID,unsigned int width,unsigned int height,unsigned int framerate)
{
    XnStatus rc;
    depthGenerators[devID].Create(ctx);
    rc = ctx.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerators[devID]);
    if (rc != XN_STATUS_OK) { printf("No depth node exists! Check your XML."); return 0; }

    imageGenerators[devID].Create(ctx);
    rc = ctx.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerators[devID]);
    if (rc != XN_STATUS_OK) { printf("No image node exists! Check your XML."); return 0; }

    XnMapOutputMode mapMode;
    mapMode.nXRes = XN_VGA_X_RES;//width;
    mapMode.nYRes = XN_VGA_Y_RES;//height;
    mapMode.nFPS = framerate;
    if (depthGenerators[devID]) { depthGenerators[devID].SetMapOutputMode(mapMode); }
    if (imageGenerators[devID]) { imageGenerators[devID].SetMapOutputMode(mapMode); }

    ctx.StartGeneratingAll();


    depthGenerators[devID].GetMetaData(depthGeneratorsMetaData[devID]);
    imageGenerators[devID].GetMetaData(imageGeneratorsMetaData[devID]);

    // Hybrid mode isn't supported in this sample
if ( ( imageGeneratorsMetaData[devID].FullXRes() != depthGeneratorsMetaData[devID].FullXRes()) ||
     (imageGeneratorsMetaData[devID].FullYRes() != depthGeneratorsMetaData[devID].FullYRes())  )
   {
      printf ("The device depth and image resolution must be equal!\n");
      return 0;
   }


// RGB is the only image format supported.
if (imageGeneratorsMetaData[devID].PixelFormat() != XN_PIXEL_FORMAT_RGB24)
  {
    printf("The device image format must be RGB24\n");
    return 0;
  }

   imageGenerators[devID].GetMirrorCap().SetMirror(false);
   depthGenerators[devID].GetMirrorCap().SetMirror(false);

 return 1;
}

int destroyOpenNI1Device(int devID) { return 0; }

int snapOpenNI1Frames(int devID)
{
  XnStatus rc = XN_STATUS_OK;
 // Read a new frame
 rc = ctx.WaitAnyUpdateAll();
 if (rc != XN_STATUS_OK) { printf("Read failed: %s\n", xnGetStatusString(rc)); return 0; }

  depthGenerators[devID].GetMetaData(depthGeneratorsMetaData[devID]);
  imageGenerators[devID].GetMetaData(imageGeneratorsMetaData[devID]);

 return 1;
}

//Color Frame getters
int getOpenNI1ColorWidth(int devID) { return imageGeneratorsMetaData[devID].FullXRes(); }
int getOpenNI1ColorHeight(int devID) { return imageGeneratorsMetaData[devID].FullYRes(); }
int getOpenNI1ColorDataSize(int devID) { return getOpenNI1ColorWidth(devID)*getOpenNI1ColorHeight(devID)*3; }
int getOpenNI1ColorChannels(int devID) { return 3; }
int getOpenNI1ColorBitsPerPixel(int devID) { return 8; }
char * getOpenNI1ColorPixels(int devID) { return (char*) imageGeneratorsMetaData[devID].RGB24Data(); }


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
    return (double) pixelSize;
}



//Depth Frame getters
int getOpenNI1DepthWidth(int devID) { return depthGeneratorsMetaData[devID].FullXRes(); }
int getOpenNI1DepthHeight(int devID) { return depthGeneratorsMetaData[devID].FullYRes(); }
int getOpenNI1DepthDataSize(int devID) { return getOpenNI1DepthWidth(devID)*getOpenNI1DepthHeight(devID); }
int getOpenNI1DepthChannels(int devID) { return 1; }
int getOpenNI1DepthBitsPerPixel(int devID) { return 16; }
short * getOpenNI1DepthPixels(int devID) { return (short*) depthGeneratorsMetaData[devID].Data(); }

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
    return (double) pixelSize;
}

