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

using namespace DepthSense;
using namespace std;



// Open CV vars
IplImage
  *g_depthImage=NULL,
  *g_videoImage=NULL; // initialized in main, used in CBs
CvSize
  g_szDepth=cvSize(160,120), // QQVGA
  g_szVideo=cvSize(640,480); //VGA

bool g_saveImageFlag=false, g_saveDepthFlag=false;

Context g_context;
DepthNode g_dnode;
ColorNode g_cnode;
AudioNode g_anode;

uint32_t g_aFrames = 0;
uint32_t g_cFrames = 0;
uint32_t g_dFrames = 0;

clock_t g_fTime;

bool g_bDeviceFound = false;

ProjectionHelper* g_pProjHelper = NULL;
StereoCameraParameters g_scp;


// From SoftKinetic
// convert a YUY2 image to RGB

void yuy2rgb(unsigned char *dst, const unsigned char *src, const int width, const int height) {
  int x, y;
  const int width2 = width * 2;
  const int width4 = width * 3;
  const unsigned char *src1 = src;
  unsigned char *dst1 = dst;

  for (y=0; y<height; y++) {
    for (x=0; x<width; x+=2) {
      int x2=x*2;
      int y1  = src1[x2  ];
      int y2  = src1[x2+2];
      int u   = src1[x2+1] - 128;
      int v   = src1[x2+3] - 128;
      int uvr = (          15748 * v) / 10000;
      int uvg = (-1873 * u - 4681 * v) / 10000;
      int uvb = (18556 * u          ) / 10000;

      int x4=x*3;
      int r1 = y1 + uvr;
      int r2 = y2 + uvr;
      int g1 = y1 + uvg;
      int g2 = y2 + uvg;
      int b1 = y1 + uvb;
      int b2 = y2 + uvb;

      dst1[x4+0] = (b1 > 255) ? 255 : ((b1 < 0) ? 0 : b1);
      dst1[x4+1] = (g1 > 255) ? 255 : ((g1 < 0) ? 0 : g1);
      dst1[x4+2] = (r1 > 255) ? 255 : ((r1 < 0) ? 0 : r1);
      //dst1[x4+3] = 255;

      dst1[x4+3] = (b2 > 255) ? 255 : ((b2 < 0) ? 0 : b2);
      dst1[x4+4] = (g2 > 255) ? 255 : ((g2 < 0) ? 0 : g2);
      dst1[x4+5] = (r2 > 255) ? 255 : ((r2 < 0) ? 0 : r2);
    }
    src1 += width2;
    dst1 += width4;
  }
}

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

    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&w,&h);

    yuy2rgb((unsigned char *)g_videoImage->imageData,data.colorMap,w,h);

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
  //printf("Z#%u: %d\n",g_dFrames,data.vertices.size());

    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&w,&h);

    int count=0; // DS data index
    if (data.depthMapFloatingPoint!=0)// just in case !
      for (int i=0; i<h; i++)
    for (int j=0; j<w; j++) {
          // some arbitrary scaling to make this visible
      float val = data.depthMapFloatingPoint[count++];
      if (!g_saveImageFlag && !g_saveDepthFlag) val*=150;
      if (val<0) val=255; // catch the saturated points
      cvSet2D(g_depthImage,i,j,cvScalar(val));
    }

    g_dFrames++;

    /*
    // Quit the main loop after 200 depth frames received
    if (g_dFrames == 20) {
      printf("Quitting main loop after MAX frames\n");
        g_context.quit();
    }
    */

    /* OpenCV display - this will slow stuff down, should be in thread*/

    cvShowImage("Video",g_videoImage);
    cvShowImage("Depth",g_depthImage);

    if (g_saveImageFlag || g_saveDepthFlag) { // save a timestamped image pair; synched by depth image time
      char filename[100];
      g_fTime = clock();
      sprintf(filename,"df%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      cvSaveImage(filename,g_depthImage);
      sprintf(filename,"vf%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      if (g_saveImageFlag)
    cvSaveImage(filename,g_videoImage);
    }

    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27) {
      printf("Quitting main loop from OpenCV\n");
        g_context.quit();
    } else
      if (key=='W') g_saveImageFlag = !g_saveImageFlag;
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
    config.frameFormat = FRAME_FORMAT_QQVGA;
    config.framerate = 60;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    //    g_dnode.setEnableVertices(true);
    g_dnode.setEnableDepthMapFloatingPoint(true);



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


    config.frameFormat = FRAME_FORMAT_QQVGA;
    config.compression = COMPRESSION_TYPE_MJPEG;

    //config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    //config.framerate = 25;

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

        printf("Found %u nodes\n",na.size());

        for (int n = 0; n < (int)na.size();n++)
            configureNode(na[n]);
    }

    /* Some OpenCV init; make windows and buffers to display the data */



    // VGA format color image
    g_videoImage=cvCreateImage(g_szVideo,IPL_DEPTH_8U,3);
    if (g_videoImage==NULL)
      { printf("Unable to create video image buffer\n"); exit(0); }

    // QVGA format depth image
    g_depthImage=cvCreateImage(g_szDepth,IPL_DEPTH_8U,1);
    if (g_depthImage==NULL)
      { printf("Unable to create depth image buffer\n"); exit(0);}

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
