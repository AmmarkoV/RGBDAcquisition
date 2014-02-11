#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include <DepthSense.hxx>

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
int main(int argc, char* argv[]);
