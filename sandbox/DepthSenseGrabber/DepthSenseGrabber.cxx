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

bool exportJPG = false;

// Resolution type: 0: QQVGA; 1: QVGA; 2:VGA; 3:WXGA_H; 4:NHD
int resDepthType = 1;
int resColorType = 2;
//int resDepthX,resDepthY,resColorX,resColorY;
int frameRateDepth = 30;
int frameRateColor = 30;

int resDepthX = formatResX(resDepthType), resDepthY= formatResY(resDepthType);
int resColorX = formatResX(resColorType), resColorY= formatResY(resColorType);

int timeStamp;


unsigned int depthFrameCount, colorFrameCount;

//printf("%i,%i\n",resDepthX,resDepthY);


//int pixelsDepth[10];

//vector<int> pixelsDepth(resDepthX*resDepthY);
//vector<int> pixelsColor(resColorX*resColorY);

// Open CV vars
IplImage
*g_depthImage=NULL,
 *g_colorImage=NULL; // initialized in main, used in CBs
CvSize
//g_szDepth=cvSize(160,120), // QQVGA
g_szDepth=cvSize(resDepthX,resDepthY), // QVGA
g_szColor=cvSize(resColorX,resColorY); //VGA

bool g_saveImageFlag=false, g_saveDepthFlag=false;

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

char fileNameColor[50];
char fileNameDepth[50];




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
    //printf("C#%u: %d\n",g_cFrames,data.colorMap.size());

    timeStamp = (int) (((float)(1000*clock()))/CLOCKS_PER_SEC);

    int32_t width, height;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&width,&height);

    uint8_t pixelsRGB[3*width*height];

    int count=0; // DS data index
    if (data.colorMap!=0)// just in case !
        for (int i=0; i<height; i++)
            for (int j=0; j<width; j++)
            {
                pixelsRGB[3*count] = data.colorMap[3*count+2];
                pixelsRGB[3*count+1] = data.colorMap[3*count+1];
                pixelsRGB[3*count+2] = data.colorMap[3*count];
                cvSet2D(g_colorImage,i,j,cvScalar(data.colorMap[3*count],data.colorMap[3*count+1],data.colorMap[3*count+2])); //BGR format
                count++;
            }

    g_cFrames++;

    if (g_saveImageFlag)
    {
        //g_fTime = clock();
        if (exportJPG)
        {
            //sprintf(fileNameColor,"colorFrame_%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
            sprintf(fileNameColor,"colorFrame_%05u.jpg",colorFrameCount);
            cvSaveImage(fileNameColor,g_colorImage);
        }
        else
        {
            //sprintf(fileNameColor,"colorFrame_%d.%d.pnm",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
            sprintf(fileNameColor,"colorFrame_%05u.pnm",colorFrameCount);
            saveRawColorFrame(fileNameColor, pixelsRGB, width, height, timeStamp);
        }
        colorFrameCount++;
    }
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
    int32_t width, height;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&width,&height);
    int val = 0;
    unsigned short pixelsDepth[width*height];
    int count=0; // DS data index
    if (data.depthMap!=0)// just in case !
        for (int i=0; i<height; i++)
            for (int j=0; j<width; j++)
            {
                val = data.depthMap[count];
                cvSet2D(g_depthImage,i,j,cvScalar(val));
                pixelsDepth[count] = (unsigned short) (val);
                count++;
            }

    g_dFrames++;

    /* OpenCV display - this will slow stuff down, should be in thread*/

    cvShowImage("Color",g_colorImage);
    cvShowImage("Depth",g_depthImage);

    if (g_saveImageFlag || g_saveDepthFlag)   // save a timestamped image pair; synched by depth image time
    {
        if (exportJPG)
        {
            //(fileNameDepth,"depthFrame_%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
            sprintf(fileNameDepth,"depthFrame_%05u.jpg",depthFrameCount);
            cvSaveImage(fileNameDepth,g_depthImage);
        }
        else
        {
            //sprintf(fileNameDepth,"depthFrame_%d.%d.pnm",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
            sprintf(fileNameDepth,"depthFrame_%05u.pnm",depthFrameCount);
            saveRawDepthFrame(fileNameDepth, pixelsDepth, width, height, timeStamp);
        }
        depthFrameCount++;
    }

    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27)
    {
        printf("Quitting main loop from OpenCV\n");
        g_context.quit();
    }
    else if (key=='W') g_saveImageFlag = !g_saveImageFlag;
    else if (key=='w') g_saveDepthFlag = !g_saveDepthFlag;
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
    config.frameFormat = formatName(resDepthType);
    config.framerate = frameRateDepth;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    g_dnode.setEnableDepthMap(true);

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
    config.frameFormat = formatName(resColorType);
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
    g_colorImage=cvCreateImage(g_szColor,IPL_DEPTH_8U,3);
    if (g_colorImage==NULL)
    {
        printf("Unable to create color image buffer\n");
        exit(0);
    }

    // QVGA format depth image
    g_depthImage=cvCreateImage(g_szDepth,IPL_DEPTH_8U,1);
    if (g_depthImage==NULL)
    {
        printf("Unable to create depth image buffer\n");
        exit(0);
    }

    printf("dml@Fordham version of DS ConsoleDemo. June 2013.\n");
    printf("Click onto in image for commands. ESC to exit.\n");
    printf("Use \'W\' to toggle dumping of depth and visual images.\n");
    printf("Use \'w\' to toggle dumping of depth images only.\n\n");
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
