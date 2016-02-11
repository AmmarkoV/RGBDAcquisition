#ifndef NITE2_H_INCLUDED
#define NITE2_H_INCLUDED


#include <OpenNI.h>
#include <PS1080.h>

using namespace openni;

#include "../tools/ImagePrimitives/skeleton.h"

int registerSkeletonPointingDetectedEvent(int devID,void * callback);
int registerSkeletonDetectedEvent(int devID,void * callback);

int startNite2(int maxVirtualSkeletonTrackers);
int createNite2Device(int devID,openni::Device * device);
int destroyNite2Device(int devID);

int stopNite2();
int loopNite2(int devID,unsigned int frameNumber);

unsigned short  * getNite2DepthFrame(int devID);
int getNite2DepthHeight(int devID);
int getNite2DepthWidth(int devID);

#endif // NITE2_H_INCLUDED
