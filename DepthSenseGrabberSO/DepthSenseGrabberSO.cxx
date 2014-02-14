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

#include <DepthSense.hxx>

#include "DepthSenseGrabberSO.hxx"
#include "../shared/ConversionTools.hxx"

using namespace DepthSense;
using namespace std;

int waitSecondsBeforeGrab = 0;

bool interpolateDepthFlag = 1;

bool dispColorAcqFlag = 0;
bool dispDepthAcqFlag = 0;
bool dispColorSyncFlag = 0;
bool dispDepthSyncFlag = 0;

bool saveColorAcqFlag = 0;
bool saveDepthAcqFlag = 1;
bool saveColorSyncFlag = 1;
bool saveDepthSyncFlag = 0;

//int widthQVGA,heightQVGA,widthColor,heightColor;
int frameRateDepth = 30;
int frameRateColor = 30;

const int widthQVGA = 320, heightQVGA = 240;
const int widthVGA = 640, heightVGA = 480;
const int widthWXGA = 1280, heightWXGA = 720;
const int widthNHD = 640, heightNHD = 360;

const int nPixelsQVGA = widthQVGA*heightQVGA;
const int nPixelsVGA = widthVGA*heightVGA;
const int nPixelsWXGA = widthWXGA*heightWXGA;
const int nPixelsNHD = widthNHD*heightNHD;

// Acquired data
uint16_t pixelsDepthAcqQVGA[nPixelsQVGA];
uint8_t pixelsColorAcqVGA[3*nPixelsVGA];
uint8_t pixelsColorAcqWXGA[3*nPixelsWXGA];
uint8_t pixelsColorAcqNHD[3*nPixelsNHD];

// UVmap-processed frames
uint8_t pixelsColorSyncQVGA[3*nPixelsQVGA];
uint16_t pixelsDepthSyncQVGA[nPixelsQVGA];
uint16_t pixelsDepthSyncVGA[nPixelsVGA];
uint16_t pixelsDepthSyncWXGA[nPixelsWXGA];
uint16_t pixelsDepthSyncNHD[nPixelsNHD];

// Interpolated frames
uint8_t pixelsColorSyncVGA[3*nPixelsVGA];
uint16_t pixelsDepthAcqVGA[nPixelsVGA];


FrameFormat frameFormatDepth = FRAME_FORMAT_QVGA; // Depth QVGA
const int nPixelsDepthAcq = nPixelsQVGA;
uint16_t* pixelsDepthAcq = pixelsDepthAcqQVGA;





// Color VGA
FrameFormat frameFormatColor = FRAME_FORMAT_VGA;
const int widthColor = widthVGA, heightColor = heightVGA, nPixelsColorAcq = nPixelsVGA;
uint8_t* pixelsColorAcq = pixelsColorAcqVGA;
uint16_t* pixelsDepthSync = pixelsDepthSyncVGA;

/*
// Color WXGA
FrameFormat frameFormatColor = FRAME_FORMAT_WXGA_H;
const int widthColor = widthWXGA, heightColor = heightWXGA, nPixelsColorAcq = nPixelsWXGA;
uint8_t* pixelsColorAcq = pixelsColorAcqWXGA;
uint16_t* pixelsDepthSync = pixelsDepthSyncWXGA;
*/

/*
// Color NHD
FrameFormat frameFormatColor = FRAME_FORMAT_NHD;
const int widthColor = widthNHD, heightColor = heightNHD, nPixelsColorAcq = nPixelsNHD;
uint8_t* pixelsColorAcq = pixelsColorAcqNHD;
uint16_t* pixelsDepthSync = pixelsDepthSyncNHD;
*/

const uint16_t noDepthDefault = 65535;
const uint16_t noDepthThreshold = 2000;

uint8_t noDepthBGR[3] = {255,255,255};


int colorPixelInd, colorPixelRow, colorPixelCol;
int debugInt;

UV uvMapAcq[nPixelsQVGA];
UV uvMapVGA[nPixelsVGA];

int timeStamp;
clock_t clockStartGrab;

unsigned int frameCount;

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

char fileNameColorAcq[50];
char fileNameDepthAcq[50];
char fileNameColorSync[50];
char fileNameDepthSync[50];

char baseNameColorAcq[20] = "colorAcqFrame_0_";
char baseNameDepthAcq[20] = "depthFrame_0_";
char baseNameColorSync[20] = "colorFrame_0_";
char baseNameDepthSync[20] = "depthSyncFrame_0_";


/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
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
    for (int currentPixelInd = 0; currentPixelInd < nPixelsColorAcq; currentPixelInd++)
    {
        // Reinitialize synchronized depth
        pixelsDepthSync[currentPixelInd] = noDepthDefault;
        pixelsColorAcq[3*currentPixelInd] = data.colorMap[3*currentPixelInd+2];
        pixelsColorAcq[3*currentPixelInd+1] = data.colorMap[3*currentPixelInd+1];
        pixelsColorAcq[3*currentPixelInd+2] = data.colorMap[3*currentPixelInd];
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
    timeStamp = (int) (((float)(1000*(clock()-clockStartGrab)))/CLOCKS_PER_SEC);

    /*
    for (int currentPixelInd = 0; currentPixelInd < nPixelsDepthVGA; currentPixelInd++)
    {
        pixelsDepthSyncWXGA[currentPixelInd] = noDepthDefault;
    }
    */

    // Initialize raw depth and UV maps
    for (int currentPixelInd = 0; currentPixelInd < nPixelsDepthAcq; currentPixelInd++)
    {
        pixelsDepthSyncQVGA[currentPixelInd] = noDepthDefault;
        uvMapAcq[currentPixelInd] = data.uvMap[currentPixelInd];
        if (data.depthMap[currentPixelInd] < noDepthThreshold)
            pixelsDepthAcq[currentPixelInd] = data.depthMap[currentPixelInd];
        else
            pixelsDepthAcq[currentPixelInd] = noDepthDefault;
        if (interpolateDepthFlag == 0)
        {
            pixelsColorSyncQVGA[3*currentPixelInd] = noDepthBGR[2];
            pixelsColorSyncQVGA[3*currentPixelInd+1] = noDepthBGR[1];
            pixelsColorSyncQVGA[3*currentPixelInd+2] = noDepthBGR[0];
        }
    }

    if (interpolateDepthFlag)
    {
        rescaleDepth(pixelsDepthAcq, pixelsDepthAcqVGA, widthQVGA, heightQVGA, widthVGA, heightVGA);
        rescaleUV(uvMapAcq, uvMapVGA, widthQVGA, heightQVGA, widthVGA, heightVGA);
        for (int currentPixelInd = 0; currentPixelInd < nPixelsVGA; currentPixelInd++)
        {
            uvToColorPixelInd(uvMapVGA[currentPixelInd], widthColor, heightColor, &colorPixelInd, &colorPixelRow, &colorPixelCol);
            if (colorPixelInd == -1) {
                pixelsColorSyncVGA[3*currentPixelInd] = noDepthBGR[2];
                pixelsColorSyncVGA[3*currentPixelInd+1] = noDepthBGR[1];
                pixelsColorSyncVGA[3*currentPixelInd+2] = noDepthBGR[0];
            }
            else
            {
                pixelsDepthSync[colorPixelInd] = pixelsDepthAcqVGA[currentPixelInd];
                pixelsColorSyncVGA[3*currentPixelInd] = pixelsColorAcq[3*colorPixelInd];
                pixelsColorSyncVGA[3*currentPixelInd+1] = pixelsColorAcq[3*colorPixelInd+1];
                pixelsColorSyncVGA[3*currentPixelInd+2] = pixelsColorAcq[3*colorPixelInd+2];
            }
        }
    }
    else
    {
        for (int currentPixelInd = 0; currentPixelInd < nPixelsQVGA; currentPixelInd++)
        {
            uvToColorPixelInd(uvMapAcq[currentPixelInd], widthColor, heightColor, &colorPixelInd, &colorPixelRow, &colorPixelCol);
            if (colorPixelInd != -1) {
                pixelsDepthSync[colorPixelInd] = pixelsDepthAcq[currentPixelInd];
                pixelsColorSyncQVGA[3*currentPixelInd] = pixelsColorAcq[3*colorPixelInd];
                pixelsColorSyncQVGA[3*currentPixelInd+1] = pixelsColorAcq[3*colorPixelInd+1];
                pixelsColorSyncQVGA[3*currentPixelInd+2] = pixelsColorAcq[3*colorPixelInd+2];
            }
        }
    }

    g_dFrames++;

    if (saveDepthAcqFlag) {
        sprintf(fileNameDepthAcq,"%s%05u.pnm",baseNameDepthAcq,frameCount);
        if (interpolateDepthFlag) saveRawDepthFrame(fileNameDepthAcq, pixelsDepthAcqVGA, widthVGA, heightVGA, timeStamp);
        else saveRawDepthFrame(fileNameDepthAcq, pixelsDepthAcq, widthQVGA, heightQVGA, timeStamp);
    }
    if (saveColorAcqFlag) {
        sprintf(fileNameColorAcq,"%s%05u.pnm",baseNameColorAcq,frameCount);
        saveRawColorFrame(fileNameColorAcq, pixelsColorAcq, widthColor, heightColor, timeStamp);
    }
    if (saveDepthSyncFlag) {
        sprintf(fileNameDepthSync,"%s%05u.pnm",baseNameDepthSync,frameCount);
        saveRawDepthFrame(fileNameDepthSync, pixelsDepthSync, widthColor, heightColor, timeStamp);
    }
    if (saveColorSyncFlag) {
        sprintf(fileNameColorSync,"%s%05u.pnm",baseNameColorSync,frameCount);
        if (interpolateDepthFlag) saveRawColorFrame(fileNameColorSync, pixelsColorSyncVGA, widthVGA, heightVGA, timeStamp);
        else saveRawColorFrame(fileNameColorSync, pixelsColorSyncQVGA, widthQVGA, heightQVGA, timeStamp);
    }
    frameCount++;

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
        g_cnode.setBrightness(0);
        g_cnode.setContrast(5);
        g_cnode.setSaturation(5);
        g_cnode.setHue(0);
        g_cnode.setGamma(3);
        g_cnode.setWhiteBalance(4650);
        g_cnode.setSharpness(5);
        g_cnode.setWhiteBalanceAuto(true);
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
        //g_context.registerNode(node); // switch this off to save bandwidth
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


    printf("dml@Fordham version of DS ConsoleDemo. June 2013.\n");
    printf("Updated Feb. 2014 (THP).\n");
    printf("Click onto in image for commands. ESC to exit.\n");
    printf("Use \'W\' or \'w\' to toggle frame dumping.\n");

    clockStartGrab = clock()+CLOCKS_PER_SEC*waitSecondsBeforeGrab;

    g_context.startNodes();

    printf("Waiting %i seconds before grabbing...\n",waitSecondsBeforeGrab);
    while (clock() < clockStartGrab);
    printf("Now grabbing!\n");

    g_context.run();

    g_context.stopNodes();

    if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
    if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
    if (g_anode.isSet()) g_context.unregisterNode(g_anode);

    if (g_pProjHelper)
        delete g_pProjHelper;

    return 0;
}
