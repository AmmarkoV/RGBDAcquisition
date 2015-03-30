// DepthSenseGrabber
// http://github.com/ph4m

#ifdef _MSC_VER
#include <windows.h>
#endif
/*
#ifdef _WIN32
#ifdef DEPTHSENSEGRABBERCORE_EXPORTS
#    define DEPTHSENSEGRABBERCORE_API __declspec(dllexport)
#else
#    define DEPTHSENSEGRABBERCORE_API __declspec(dllimport)
#endif
#else
#	define DEPTHSENSEGRABBERCORE_API
#endif // WIN32
*/

#include <stdio.h>
#include <time.h>

#include <boost/thread.hpp>

//#include <sys/time.h>
//#include <unistd.h>
#include <chrono>

#include <vector>
#include <exception>

#include <DepthSense.hxx>

#include "DepthSenseGrabberCore.hxx"
#include "../shared/ConversionTools.hxx"
#include "../shared/AcquisitionParameters.hxx"

using namespace DepthSense;
using namespace std;

struct timeval timeStart, timeCurrent;
long seconds, useconds;
std::chrono::high_resolution_clock::time_point timeStart_hr, timeCurrent_hr;

//bool flagTakingSnapshot = 0;

bool usingUSB30Flag = true; // if the camera is plugged on a USB 3.0 port

int waitSecondsBeforeGrab = 1;
const int16_t confidenceThreshold = 150;

bool interpolateDepthFlag;
bool buildColorSyncFlag, buildDepthSyncFlag, buildConfidenceFlag;

int32_t  frameRateDepth = 30;
int32_t  frameRateColor = 30;

// Acquired data
uint16_t pixelsConfidenceQVGA[FORMAT_QVGA_PIXELS];
uint16_t pixelsDepthAcqQVGA[FORMAT_QVGA_PIXELS];
uint8_t pixelsColorAcqVGA[3*FORMAT_VGA_PIXELS];
uint8_t pixelsColorAcqWXGA[3*FORMAT_WXGA_PIXELS];
uint8_t pixelsColorAcqNHD[3*FORMAT_NHD_PIXELS];

// UVmap-processed frames
uint8_t pixelsColorSyncQVGA[3*FORMAT_QVGA_PIXELS];
uint16_t pixelsDepthSyncQVGA[FORMAT_QVGA_PIXELS];
uint16_t pixelsDepthSyncVGA[FORMAT_VGA_PIXELS];
uint16_t pixelsDepthSyncWXGA[FORMAT_WXGA_PIXELS];
uint16_t pixelsDepthSyncNHD[FORMAT_NHD_PIXELS];

// Interpolated frames
uint8_t pixelsColorSyncVGA[3*FORMAT_VGA_PIXELS];
uint16_t pixelsDepthAcqVGA[FORMAT_VGA_PIXELS];


FrameFormat frameFormatDepth = FRAME_FORMAT_QVGA; // Depth QVGA
const int nPixelsDepthAcq = FORMAT_QVGA_PIXELS;
uint16_t* pixelsDepthAcq = pixelsDepthAcqQVGA;


int deltaPixelsIndAround[8] = {-641,-640,-639,-1,1,639,640,641};
bool* hasData;

// Color map configuration, comment out undesired parameters

FrameFormat frameFormatColor;
int widthColor, heightColor, nPixelsColorAcq;
uint8_t* pixelsColorAcq;
uint16_t* pixelsDepthSync;

// Snapshot data
uint16_t* pixelsDepthAcqVGASnapshot;
uint8_t* pixelsColorSyncVGASnapshot;
uint16_t* pixelsDepthAcqQVGASnapshot;
uint8_t* pixelsColorSyncQVGASnapshot;
uint8_t* pixelsColorAcqSnapshot;
uint16_t* pixelsDepthSyncSnapshot;
uint16_t* pixelsConfidenceQVGASnapshot;


const uint16_t noDepthDefault = 65535;
const uint16_t noDepthThreshold = 2000;
const uint16_t deltaDepthSync = 132; // DS325

uint8_t noDepthBGR[3] = {0,0,0};


int colorPixelInd, colorPixelRow, colorPixelCol;
int debugInt;

UV uvMapAcq[FORMAT_QVGA_PIXELS];
UV uvMapVGA[FORMAT_VGA_PIXELS];

double timeStampSeconds;
int timeStamp;
clock_t clockStartGrab;

int frameCount = -1;

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
        hasData[currentPixelInd] = 0;
        pixelsDepthSync[currentPixelInd] = noDepthDefault;
        pixelsColorAcq[3*currentPixelInd]   = data.colorMap[3*currentPixelInd+2];
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
    // Initialize raw depth and UV maps
    for (int currentPixelInd = 0; currentPixelInd < nPixelsDepthAcq; currentPixelInd++)
    {
        if (buildConfidenceFlag) pixelsConfidenceQVGA[currentPixelInd] = data.confidenceMap[currentPixelInd];
        pixelsDepthSyncQVGA[currentPixelInd] = noDepthDefault;
        uvMapAcq[currentPixelInd] = data.uvMap[currentPixelInd];
        if (data.confidenceMap[currentPixelInd] > confidenceThreshold)
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
        rescaleMap(pixelsDepthAcq, pixelsDepthAcqVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT);
        rescaleMap(uvMapAcq, uvMapVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT);
        for (int currentPixelRow = 0; currentPixelRow < FORMAT_VGA_HEIGHT; currentPixelRow++) {
            for (int currentPixelCol = 0; currentPixelCol < FORMAT_VGA_WIDTH; currentPixelCol++) {
                int currentPixelInd = currentPixelRow*FORMAT_VGA_WIDTH+currentPixelCol;
                int currentPixelIndQVGA = currentPixelRow/2*FORMAT_QVGA_WIDTH+currentPixelCol/2;
                uvToColorPixelInd(uvMapVGA[currentPixelInd], widthColor, heightColor, &colorPixelInd, &colorPixelRow, &colorPixelCol);
                if (colorPixelInd == -1 || data.confidenceMap[currentPixelIndQVGA] < confidenceThreshold) {
                    pixelsColorSyncVGA[3*currentPixelInd] = noDepthBGR[2];
                    pixelsColorSyncVGA[3*currentPixelInd+1] = noDepthBGR[1];
                    pixelsColorSyncVGA[3*currentPixelInd+2] = noDepthBGR[0];
                }
                else
                {
                    hasData[colorPixelInd] = 1;
                    pixelsDepthSync[colorPixelInd] = pixelsDepthAcqVGA[currentPixelInd] + deltaDepthSync;
                    pixelsColorSyncVGA[3*currentPixelInd] = pixelsColorAcq[3*colorPixelInd];
                    pixelsColorSyncVGA[3*currentPixelInd+1] = pixelsColorAcq[3*colorPixelInd+1];
                    pixelsColorSyncVGA[3*currentPixelInd+2] = pixelsColorAcq[3*colorPixelInd+2];
                }
            }
        }
        for (int currentRow = 1; currentRow < FORMAT_VGA_HEIGHT-1; currentRow++) {
            for (int currentCol = 1; currentCol < FORMAT_VGA_WIDTH-1; currentCol++) {
                int currentPixelInd = currentRow*FORMAT_VGA_WIDTH+currentCol;
                int countValidAround = 0;
                uint16_t depthValidAround = 0;
                if (hasData[currentPixelInd] == 0) {
                    for (int indDeltaPixel = 0; indDeltaPixel < 8; indDeltaPixel++) {
                        if (hasData[currentPixelInd+deltaPixelsIndAround[indDeltaPixel]]) {
                            countValidAround++;
                            depthValidAround = depthValidAround + pixelsDepthSync[currentPixelInd+deltaPixelsIndAround[indDeltaPixel]];
                        }
                    }
                    if (countValidAround > 1) {
                        pixelsDepthSync[currentPixelInd] = depthValidAround / countValidAround;
                    }
                }
            }
        }
    }
    else
    {
        for (int currentPixelInd = 0; currentPixelInd < FORMAT_QVGA_PIXELS; currentPixelInd++)
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

    // Saving snapshot...
    //memcpy(pixelsDepthAcqQVGASnapshot, pixelsDepthAcqQVGA, FORMAT_QVGA_PIXELS*sizeof(uint16_t));
    //memcpy(pixelsDepthAcqVGASnapshot, pixelsDepthAcqVGA, FORMAT_VGA_PIXELS*sizeof(uint16_t));
    //memcpy(pixelsColorAcqSnapshot, pixelsColorAcq, 3*nPixelsColorAcq*sizeof(uint8_t));
    //memcpy(pixelsConfidenceQVGASnapshot, pixelsConfidenceQVGA, FORMAT_QVGA_PIXELS*sizeof(uint16_t));
    if (buildColorSyncFlag) {
        if (interpolateDepthFlag) {
            memcpy(pixelsColorSyncVGASnapshot, pixelsColorSyncVGA, 3*FORMAT_VGA_PIXELS*sizeof(uint8_t));
        } else {
            memcpy(pixelsColorSyncQVGASnapshot, pixelsColorSyncQVGA, 3*FORMAT_QVGA_PIXELS*sizeof(uint8_t));
        }
    }
    if (buildDepthSyncFlag) {
        memcpy(pixelsDepthSyncSnapshot, pixelsDepthSync, nPixelsColorAcq*sizeof(uint16_t));
    }


    g_dFrames++;

    //gettimeofday(&timeCurrent, NULL);
	//seconds = timeCurrent.tv_sec - timeStart.tv_sec;
	//useconds = timeCurrent.tv_usec - timeStart.tv_usec;
	//timeStamp = (int)(seconds * 1000 + useconds / 1000.0 + 0.5);
	timeCurrent_hr = std::chrono::high_resolution_clock::now();
	timeStamp = (int) std::chrono::duration_cast<std::chrono::milliseconds>(timeCurrent_hr - timeStart_hr).count();

    frameCount++;

}

uint16_t* getPixelsDepthAcqQVGA() {
    return pixelsDepthAcqQVGA;
    //return pixelsDepthAcqQVGASnapshot;
}
uint16_t* getPixelsDepthAcqVGA() {
    return pixelsDepthAcqVGA;
    //return pixelsDepthAcqVGASnapshot;
}
uint8_t* getPixelsColorsAcq() {
    return pixelsColorAcq;
}
uint16_t* getPixelsDepthSync() {
    //return pixelsDepthSync;
    return pixelsDepthSyncSnapshot;
}
uint8_t* getPixelsColorSyncVGA() {
    //return pixelsColorSyncVGA;
    return pixelsColorSyncVGASnapshot;
}
uint8_t* getPixelsColorSyncQVGA() {
    //return pixelsColorSyncQVGA;
    return pixelsColorSyncQVGASnapshot;
}
uint16_t* getPixelsConfidenceQVGA() {
    return pixelsConfidenceQVGA;
}

int getTimeStamp() {
    return timeStamp;
}

int getFrameCount() {
    return frameCount;
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
    DepthNode::Configuration configRef(frameFormatDepth, frameRateDepth, DepthNode::CAMERA_MODE_CLOSE_MODE, true);
    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = frameFormatDepth;
    config.framerate = frameRateDepth;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    g_dnode.setEnableDepthMap(true);
    g_dnode.setEnableUvMap(true);
    g_dnode.setEnableConfidenceMap(true);

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
        printf("Configuring depth node\n");
        g_dnode = node.as<DepthNode>();
        configureDepthNode();
        g_context.registerNode(node);
    }

    if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
        printf("Configuring color node\n");
        g_cnode = node.as<ColorNode>();
        configureColorNode();
        g_context.registerNode(node);
    }

    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        printf("Configuring audio node\n");
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        if (usingUSB30Flag != 1) g_context.registerNode(node); // switch this off to save bandwidth
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
void capture()
{

    printf("Starting capture\n");
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


    printf("DepthSenseGrabber, Feb. 2014. (thp@pham.in)\n");

    clockStartGrab = clock()+CLOCKS_PER_SEC*waitSecondsBeforeGrab;

    g_context.startNodes();

    printf("Waiting %i seconds before grabbing...\n",waitSecondsBeforeGrab);
    while (clock() < clockStartGrab);
    printf("Now grabbing!\n");

	//gettimeofday(&timeStart, NULL);
	timeStart_hr = std::chrono::high_resolution_clock::now();

    g_context.run();

}

void start_capture(int flagColorFormat,
                   bool interpolateDepthFlag_in,
                   bool buildColorSyncFlag_in, bool buildDepthSyncFlag_in, bool buildConfidenceFlag_in)
{
    interpolateDepthFlag = interpolateDepthFlag_in;
    buildColorSyncFlag = buildColorSyncFlag_in;
    buildDepthSyncFlag = buildDepthSyncFlag_in;
    buildConfidenceFlag = buildConfidenceFlag_in;

    switch (flagColorFormat) {
        case FORMAT_VGA_ID:
            frameFormatColor = FRAME_FORMAT_VGA;
            widthColor = FORMAT_VGA_WIDTH;
            heightColor = FORMAT_VGA_HEIGHT;
            nPixelsColorAcq = FORMAT_VGA_PIXELS;
            pixelsColorAcq = pixelsColorAcqVGA;
            pixelsDepthSync = pixelsDepthSyncVGA;
            break;
        case FORMAT_WXGA_ID:
            frameFormatColor = FRAME_FORMAT_WXGA_H;
            widthColor = FORMAT_WXGA_WIDTH;
            heightColor = FORMAT_WXGA_HEIGHT;
            nPixelsColorAcq = FORMAT_WXGA_PIXELS;
            pixelsColorAcq = pixelsColorAcqWXGA;
            pixelsDepthSync = pixelsDepthSyncWXGA;
            break;
        case FORMAT_NHD_ID:
            frameFormatColor = FRAME_FORMAT_NHD;
            widthColor = FORMAT_NHD_WIDTH;
            heightColor = FORMAT_NHD_HEIGHT;
            nPixelsColorAcq = FORMAT_NHD_PIXELS;
            pixelsColorAcq = pixelsColorAcqNHD;
            pixelsDepthSync = pixelsDepthSyncNHD;
            break;
        default:
            printf("Unknown flagColorFormat");
            exit(EXIT_FAILURE);
    }

    hasData = (bool*) malloc(nPixelsColorAcq*sizeof(bool));
    // Snapshot data
    pixelsDepthAcqVGASnapshot    = (uint16_t*) malloc(FORMAT_VGA_PIXELS*sizeof(uint16_t));
    pixelsColorSyncVGASnapshot   = (uint8_t*) malloc(3*FORMAT_VGA_PIXELS*sizeof(uint8_t));
    pixelsDepthAcqQVGASnapshot   = (uint16_t*) malloc(FORMAT_QVGA_PIXELS*sizeof(uint16_t));
    pixelsColorSyncQVGASnapshot  = (uint8_t*) malloc(3*FORMAT_QVGA_PIXELS*sizeof(uint8_t));
    pixelsColorAcqSnapshot       = (uint8_t*) malloc(3*nPixelsColorAcq*sizeof(uint8_t));
    pixelsDepthSyncSnapshot      = (uint16_t*) malloc(nPixelsColorAcq*sizeof(uint16_t));
    pixelsConfidenceQVGASnapshot = (uint16_t*) malloc(FORMAT_QVGA_PIXELS*sizeof(uint16_t));


    boost::thread capture_thread(capture);
    printf("Starting capture thread\n");
}

void stop_capture()
{
    free(hasData);
    free(pixelsDepthAcqVGASnapshot);
    free(pixelsDepthAcqQVGASnapshot);
    free(pixelsColorSyncVGASnapshot);
    free(pixelsColorSyncQVGASnapshot);
    free(pixelsColorAcqSnapshot);
    free(pixelsDepthSyncSnapshot);
    free(pixelsConfidenceQVGASnapshot);

    g_context.stopNodes();

    if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
    if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
    if (g_anode.isSet()) g_context.unregisterNode(g_anode);

    if (g_pProjHelper)
        delete g_pProjHelper;
}


