#ifndef TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
#define TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED


#include "TrajectoryParser.h"


int growVirtualStreamFrames(struct VirtualObject * streamObj,unsigned int framesToAdd);
int growVirtualStreamObjectsTypes(struct VirtualStream * stream,unsigned int objectsTypesToAdd);
int growVirtualStreamObjects(struct VirtualStream * stream,unsigned int objectsToAdd);
int growVirtualStreamEvents(struct VirtualStream * stream,unsigned int eventsToAdd);
int growVirtualStreamConnectors(struct VirtualStream * stream,unsigned int connectorsToAdd);



#endif // TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
