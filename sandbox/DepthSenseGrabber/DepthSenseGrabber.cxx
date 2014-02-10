// ConsoleDemo initial modification due to Damian Lyons
// Code originally retrieved from SoftKinetic forum
// http://www.softkinetic.com/Support/Forum/tabid/110/forumid/32/threadid/1450/scope/posts/language/en-US/Default.aspx

// DepthSense 325 parameters and conversion fix - Tu-Hoa Pham (thp@pham.in)

////////////////////////////////////////////////////////////////////////////////
// SoftKinetic DepthSense SDK
//
// COPYRIGHT AND CONFIDENTIALITY NOTICE - SOFTKINETIC CONFIDENTIAL
// INFORMATION
//
// All rights reserved to SOFTKINETIC SENSORS NV (a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 15,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 341 454 - "Softkinetic
// Sensors").
//
// The source code of the SoftKinetic DepthSense Camera Drivers is
// proprietary and confidential information of Softkinetic Sensors NV.
//
// For any question about terms and conditions, please contact:
// info@softkinetic.com Copyright (c) 2002-2012 Softkinetic Sensors NV
////////////////////////////////////////////////////////////////////////////////

// Some OpenCV mods added below for viewing and saving - Damian Lyons, dlyons@fordham.edu

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include "cv.h"
#include "highgui.h"

#include <DepthSense.hxx>

#include "DepthSenseGrabber.hxx"
#include "ConversionTools.hxx"

using namespace DepthSense;
using namespace std;

bool exportJPG = 0;

bool dispColorRawFlag = 0;
bool dispDepthRawFlag = 0;
bool dispColorSyncFlag = 0;
bool dispDepthSyncFlag = 0;

bool saveColorRawFlag = 1;
bool saveDepthRawFlag = 1;
bool saveColorSyncFlag = 1;
bool saveDepthSyncFlag = 1;

// Resolution type: 0: QQVGA; 1: QVGA; 2:VGA; 3:WXGA_H; 4:NHD
int resDepthType = 1;
int resColorType = 2;
//int widthDepth,heightDepth,widthColor,heightColor;
int frameRateDepth = 30;
int frameRateColor = 30;

/*
int widthDepth = formatResX(resDepthType), heightDepth = formatResY(resDepthType);
int widthColor = formatResX(resColorType), heightColor = formatResY(resColorType);
*/

FrameFormat frameFormatDepth = FRAME_FORMAT_QVGA; const int widthDepth = 320, heightDepth = 240; // Depth QVGA

FrameFormat frameFormatColor = FRAME_FORMAT_VGA; const int widthColor = 640, heightColor= 480; // Color VGA
//FrameFormat frameFormatColor = FRAME_FORMAT_WXGA_H; const int widthColor = 1280, heightColor= 720; // Color WXGA_H
//FrameFormat frameFormatColor = FRAME_FORMAT_NHD; const int widthColor = 640, heightColor= 360; // Color NHD



const int nPixelsColor = 3*widthColor*heightColor;
const int nPixelsDepth = widthDepth*heightDepth;
uint8_t pixelsColorRaw[nPixelsColor];

const uint16_t noDepthDefault = 0; //65535;
const uint16_t noDepthThreshold = 2000; //65535;


uint8_t noDepthBGR[3];// = {255,255,255};

uint16_t pixelsDepthRaw[nPixelsDepth];
uint16_t pixelsUv[nPixelsDepth];
unsigned char pixelsColorSync[nPixelsColor];
uint16_t pixelsDepthSync[nPixelsColor];

int colorPixelInd, colorPixelRow, colorPixelCol;
UV uv;
float u,v;
int countColor, countDepth; // DS data index


int timeStamp;

int divideDepthBrightnessCV = 1;

unsigned int frameCount;

//printf("%i,%i\n",widthDepth,heightDepth);


//int pixelsDepthRaw[10];

//vector<int> pixelsDepth(widthDepth*heightDepth);
//vector<int> pixelsColor(widthColor*heightColor);

// Open CV vars
IplImage
*g_depthRawImage=NULL,
 *g_colorRawImage=NULL, // initialized in main, used in CBs
 *g_depthSyncImage=NULL, // initialized in main, used in CBs
 *g_colorSyncImage=NULL, // initialized in main, used in CBs
 *g_emptyImage=NULL; // initialized in main, used in CBs
CvSize
//g_szDepth=cvSize(160,120), // QQVGA
g_szDepthRaw=cvSize(widthDepth,heightDepth), // QVGA
g_szColorRaw=cvSize(widthColor,heightColor); //VGA

CvSize g_szDepthSync = g_szColorRaw, g_szColorSync = g_szDepthRaw;
bool g_saveFrameFlag=false;

Context g_context;
DepthNode g_dnode;
ColorNode g_cnode;
AudioNode g_anode;

uint32_t g_aFrames = 0;
uint32_t g_cFrames = 0;
uint32_t g_dFrames = 0;

bool g_bDeviceFound = false;

ProjectionHelper* g_pProjHelper = NULL;
StereoCameraParameters g_scp;

char fileNameColorRaw[50];
char fileNameDepthRaw[50];
char fileNameColorSync[50];
char fileNameDepthSync[50];

/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    //    printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    g_aFrames++;
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
/* Comments from SoftKinetic

From data you can get

::DepthSense::Pointer< uint8_t > colorMap
The color map. If captureConfiguration::compression is
DepthSense::COMPRESSION_TYPE_MJPEG, the output format is BGR, otherwise
the output format is YUY2.
 */
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{

    timeStamp = (int) (((float)(1000*clock()))/CLOCKS_PER_SEC);

    countColor = 0;

    if (data.colorMap!=0)// just in case !
        for (int i=0; i<heightColor; i++)
            for (int j=0; j<widthColor; j++)
            {
                pixelsDepthSync[countColor] = noDepthDefault;
                pixelsColorRaw[3*countColor] = data.colorMap[3*countColor+2];
                pixelsColorRaw[3*countColor+1] = data.colorMap[3*countColor+1];
                pixelsColorRaw[3*countColor+2] = data.colorMap[3*countColor];
                if (dispColorRawFlag || (saveColorRawFlag && exportJPG)) cvSet2D(g_colorRawImage,i,j,cvScalar(pixelsColorRaw[3*countColor+2],pixelsColorRaw[3*countColor+1],pixelsColorRaw[3*countColor])); //BGR format
                if (dispDepthSyncFlag || (saveDepthSyncFlag && exportJPG)) cvSet2D(g_depthSyncImage,i,j,cvScalar(pixelsDepthSync[countColor]));
                countColor++;
            }
    g_cFrames++;

}

/*----------------------------------------------------------------------------*/
// New depth sample event handler

/* From SoftKinetic

::DepthSense::Pointer< int16_t > depthMap
The depth map in fixed point format. This map represents the cartesian depth of each
pixel, expressed in millimeters. Valid values lies in the range [0 - 31999]. Saturated
pixels are given the special value 32002.
 ::DepthSense::Pointer< float > depthMapFloatingPoint
The depth map in floating point format. This map represents the cartesian depth of
each pixel, expressed in meters. Saturated pixels are given the special value -2.0.
*/

void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    timeStamp = (int) (((float)(1000*clock()))/CLOCKS_PER_SEC);
    countDepth = 0;

    if (data.depthMap!=0)// just in case !
        for (int i=0; i<heightDepth; i++)
            for (int j=0; j<widthDepth; j++)
            {
                uv = data.uvMap[countDepth];
                if (pixelsDepthRaw[countDepth] < noDepthThreshold)
                    pixelsDepthRaw[countDepth] = data.depthMap[countDepth];
                else
                    pixelsDepthRaw[countDepth] = noDepthDefault;
                uvToColorPixelInd(uv, widthColor, heightColor, &colorPixelInd, &colorPixelRow, &colorPixelCol);
                if (colorPixelInd == -1) {
                    pixelsDepthSync[colorPixelInd] = noDepthDefault;
                    pixelsColorSync[3*countDepth] = noDepthBGR[2];
                    pixelsColorSync[3*countDepth+1] = noDepthBGR[1];
                    pixelsColorSync[3*countDepth+2] = noDepthBGR[0];
                }
                else {
                    pixelsDepthSync[colorPixelInd] = data.depthMap[countDepth];
                    pixelsColorSync[3*countDepth] = pixelsColorRaw[3*colorPixelInd];
                    pixelsColorSync[3*countDepth+1] = pixelsColorRaw[3*colorPixelInd+1];
                    pixelsColorSync[3*countDepth+2] = pixelsColorRaw[3*colorPixelInd+2];
                }

                if (dispDepthRawFlag || (saveDepthRawFlag && exportJPG)) cvSet2D(g_depthRawImage,i,j,cvScalar(pixelsDepthRaw[countDepth]/divideDepthBrightnessCV));
                if (dispDepthSyncFlag || (saveDepthSyncFlag && exportJPG)) cvSet2D(g_depthSyncImage,colorPixelRow,colorPixelCol,cvScalar(pixelsDepthSync[colorPixelInd]/divideDepthBrightnessCV));
                if (dispColorSyncFlag || (saveColorSyncFlag && exportJPG)) cvSet2D(g_colorSyncImage,i,j,cvScalar(pixelsColorSync[3*countDepth+2],pixelsColorSync[3*countDepth+1],pixelsColorSync[3*countDepth])); //BGR format
                countDepth++;
            }

    g_dFrames++;

    /* OpenCV display - this will slow stuff down, should be in thread*/

    if (dispColorRawFlag) cvShowImage("Raw Color",g_colorRawImage);
    if (dispDepthRawFlag) cvShowImage("Raw Depth",g_depthRawImage);
    if (dispDepthSyncFlag) cvShowImage("Synchronized Depth",g_depthSyncImage);
    if (dispColorSyncFlag) cvShowImage("Synchronized Color",g_colorSyncImage);
    if (dispColorRawFlag+dispColorSyncFlag+dispDepthRawFlag+dispDepthSyncFlag == 0)  cvShowImage("Empty",g_emptyImage);

    if (g_saveFrameFlag)
    {
        if (exportJPG)
        {
            if (saveDepthRawFlag) {
                sprintf(fileNameDepthRaw,"depthRawFrame_%05u.jpg",frameCount);
                cvSaveImage(fileNameDepthRaw,g_depthRawImage);
            }
            if (saveColorRawFlag) {
                sprintf(fileNameColorRaw,"colorRawFrame_%05u.jpg",frameCount);
                cvSaveImage(fileNameColorRaw,g_colorRawImage);
            }
            if (saveDepthSyncFlag) {
                sprintf(fileNameDepthSync,"depthSyncFrame_%05u.jpg",frameCount);
                cvSaveImage(fileNameDepthSync,g_depthSyncImage);
            }
            if (saveColorSyncFlag) {
                sprintf(fileNameColorSync,"colorSyncFrame_%05u.jpg",frameCount);
                cvSaveImage(fileNameColorSync,g_colorSyncImage);
            }
        }
        else
        {
            if (saveDepthRawFlag) {
                sprintf(fileNameDepthRaw,"depthRawFrame_%05u.pnm",frameCount);
                saveRawDepthFrame(fileNameDepthRaw, pixelsDepthRaw, widthDepth, heightDepth, timeStamp);
            }
            if (saveColorRawFlag) {
                sprintf(fileNameColorRaw,"colorRawFrame_%05u.pnm",frameCount);
                saveRawColorFrame(fileNameColorRaw, pixelsColorRaw, widthColor, heightColor, timeStamp);
            }
            if (saveDepthSyncFlag) {
                sprintf(fileNameDepthSync,"depthSyncFrame_%05u.pnm",frameCount);
                saveRawDepthFrame(fileNameDepthSync, pixelsDepthSync, widthColor, heightColor, timeStamp);
            }
            if (saveColorSyncFlag) {
                sprintf(fileNameColorSync,"colorSyncFrame_%05u.pnm",frameCount);
                saveRawColorFrame(fileNameColorSync, pixelsColorSync, widthDepth, heightDepth, timeStamp);
            }
        }
        frameCount++;
    }



    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27)
    {
        printf("Quitting main loop from OpenCV\n");
        g_context.quit();
    }
    else if (key=='W' || key=='w') g_saveFrameFlag = !g_saveFrameFlag;
}

/*----------------------------------------------------------------------------*/
void configureAudioNode()
{
    g_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

    AudioNode::Configuration config = g_anode.getConfiguration();
    config.sampleRate = 44100;

    try
    {
        g_context.requestControl(g_anode,0);

        g_anode.setConfiguration(config);

        g_anode.setInputMixerLevel(0.5f);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}




/*----------------------------------------------------------------------------*/

void configureDepthNode()
{
    g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);
    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = frameFormatDepth;
    config.framerate = frameRateDepth;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    g_dnode.setEnableDepthMap(true);
    g_dnode.setEnableUvMap(true);

    try
    {
        g_context.requestControl(g_dnode,0);

        g_dnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("DEPTH Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("DEPTH Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("DEPTH IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("DEPTH Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("DEPTH Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("DEPTH Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("DEPTH TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
void configureColorNode()
{
    // connect new color sample handler
    g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

    ColorNode::Configuration config = g_cnode.getConfiguration();
    config.frameFormat = frameFormatColor;
    config.compression = COMPRESSION_TYPE_MJPEG; // can also be COMPRESSION_TYPE_YUY2
    config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    config.framerate = frameRateColor;

    g_cnode.setEnableColorMap(true);


    try
    {
        g_context.requestControl(g_cnode,0);

        g_cnode.setConfiguration(config);
    }
    catch (ArgumentException& e)
    {
        printf("COLOR Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("COLOR Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("COLOR IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("COLOR Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("COLOR Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("COLOR Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("COLOR TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureNode(Node node)
{
    if ((node.is<DepthNode>())&&(!g_dnode.isSet()))
    {
        g_dnode = node.as<DepthNode>();
        configureDepthNode();
        g_context.registerNode(node);
    }

    if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
        g_cnode = node.as<ColorNode>();
        configureColorNode();
        g_context.registerNode(node);
    }

    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        g_context.registerNode(node);
    }
}

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
        g_anode.unset();
    if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == g_cnode))
        g_cnode.unset();
    if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == g_dnode))
        g_dnode.unset();
    printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!g_bDeviceFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
        g_bDeviceFound = true;
    }
}

/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    g_bDeviceFound = false;
    printf("Device disconnected\n");
}

/*----------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    g_context = Context::create("localhost");

    g_context.deviceAddedEvent().connect(&onDeviceConnected);
    g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);

    // Get the list of currently connected devices
    vector<Device> da = g_context.getDevices();

    // We are only interested in the first device
    if (da.size() >= 1)
    {
        g_bDeviceFound = true;

        da[0].nodeAddedEvent().connect(&onNodeConnected);
        da[0].nodeRemovedEvent().connect(&onNodeDisconnected);

        vector<Node> na = da[0].getNodes();

        printf("Found %lu nodes\n",na.size());

        for (int n = 0; n < (int)na.size(); n++)
            configureNode(na[n]);
    }

    /* Some OpenCV init; make windows and buffers to display the data */



    // VGA format color image
    g_colorRawImage=cvCreateImage(g_szColorRaw,IPL_DEPTH_8U,3);
    if (g_colorRawImage==NULL)
    {
        printf("Unable to create color image buffer\n");
        exit(0);
    }

    // QVGA format depth image
    g_depthRawImage=cvCreateImage(g_szDepthRaw,IPL_DEPTH_8U,1);
    if (g_depthRawImage==NULL)
    {
        printf("Unable to create depth image buffer\n");
        exit(0);
    }

    // QVGA format depth color image
    g_depthSyncImage=cvCreateImage(g_szDepthSync,IPL_DEPTH_8U,1);
    if (g_depthSyncImage==NULL)
    {
        printf("Unable to create depth color image buffer\n");
        exit(0);
    }

    // QVGA format depth color image
    g_colorSyncImage=cvCreateImage(g_szColorSync,IPL_DEPTH_8U,3);
    if (g_colorSyncImage==NULL)
    {
        printf("Unable to create color depth image buffer\n");
        exit(0);
    }

    // Empty image
    g_emptyImage=cvCreateImage(g_szColorSync,IPL_DEPTH_8U,1);
    if (g_emptyImage==NULL)
    {
        printf("Unable to create empty image buffer\n");
        exit(0);
    }

    printf("dml@Fordham version of DS ConsoleDemo. June 2013.\n");
    printf("Updated Feb. 2014 (THP).\n");
    printf("Click onto in image for commands. ESC to exit.\n");
    printf("Use \'W\' or \'w\' to toggle frame dumping.\n");
    g_context.startNodes();

    g_context.run();

    g_context.stopNodes();

    if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
    if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
    if (g_anode.isSet()) g_context.unregisterNode(g_anode);

    if (g_pProjHelper)
        delete g_pProjHelper;

    return 0;
}
