#ifndef DEPTHSENSEGRABBERCORE_DEPTHSENSEGRABBERCORE_H_
#define DEPTHSENSEGRABBERCORE_DEPTHSENSEGRABBERCORE_H_

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include <DepthSense.hxx>
#include "../shared/AcquisitionParameters.hxx"


using namespace DepthSense;
using namespace std;






/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data);

/*----------------------------------------------------------------------------*/
// New color sample event handler
/* Comments from SoftKinetic

From data you can get

::DepthSense::Pointer< uint8_t > colorMap
The color map. If captureConfiguration::compression is
DepthSense::COMPRESSION_TYPE_MJPEG, the output format is BGR, otherwise
the output format is YUY2.
 */
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data);

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

void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data);

/*----------------------------------------------------------------------------*/
void configureAudioNode();


/*----------------------------------------------------------------------------*/
void configureDepthNode();

/*----------------------------------------------------------------------------*/
void configureColorNode();

/*----------------------------------------------------------------------------*/
void configureNode(Node node);

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data);

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data);

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data);
/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data);

/*----------------------------------------------------------------------------*/
//int main(int argc, char* argv[]);

void capture();

void start_capture(int flagColorFormat = FRAME_FORMAT_VGA,
                   bool interpolateDepthFlag_in = true,
                   bool buildColorSyncFlag_in = true, bool buildDepthSyncFlag_in = true,
                   bool buildConfidenceFlag_in = true);

void stop_capture();


uint16_t* getPixelsDepthAcqQVGA();
uint16_t* getPixelsDepthAcqVGA();
uint8_t* getPixelsColorsAcq();
uint16_t* getPixelsDepthSync();
uint8_t* getPixelsColorSyncVGA();
uint8_t* getPixelsColorSyncQVGA();
uint16_t* getPixelsConfidenceQVGA();
int getTimeStamp();
int getFrameCount();

#endif // DEPTHSENSEGRABBERCORE_DEPTHSENSEGRABBERCORE_H_

