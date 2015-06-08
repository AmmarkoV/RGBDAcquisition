#ifndef TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
#define TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED


#include "TrajectoryParser.h"


int growVirtualStreamFrames(struct VirtualObject * streamObj,unsigned int framesToAdd);
int growVirtualStreamObjectsTypes(struct VirtualStream * stream,unsigned int objectsTypesToAdd);
int growVirtualStreamObjects(struct VirtualStream * stream,unsigned int objectsToAdd);
int growVirtualStreamEvents(struct VirtualStream * stream,unsigned int eventsToAdd);
int growVirtualStreamConnectors(struct VirtualStream * stream,unsigned int connectorsToAdd);




static int dummy_strcasecmp_internal(char * input1, char * input2);

void listAllObjectTypeID(struct VirtualStream * stream);

ObjectIDHandler getObjectID(struct VirtualStream * stream,char * name, unsigned int * found);


ObjectTypeID getObjectTypeID(struct VirtualStream * stream,char * typeName,unsigned int * found);

char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID);

char * getModelOfObjectID(struct VirtualStream * stream,ObjectIDHandler ObjID);

int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency, unsigned char * noColor);
unsigned long getFileSize(char * filename);





int addStateToObjectID(
                               struct VirtualStream * stream ,
                               unsigned int ObjID  ,
                               unsigned int timeMilliseconds ,
                               float * coord ,
                               unsigned int coordLength ,
                               float scaleX , float scaleY ,float scaleZ ,
                               float R , float G , float B , float Alpha
                       );

int addStateToObject(
                              struct VirtualStream * stream ,
                              char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength ,
                              float scaleX , float scaleY ,float scaleZ ,
                              float R , float G , float B , float Alpha
                       );

int addStateToObjectMini(
                              struct VirtualStream * stream ,
                              char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength
                       );

int addConnectorToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * firstObject , char * secondObject ,
                                 unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                                 float scale,
                                 char * typeStr
                               );

int addObjectToVirtualStream(
                              struct VirtualStream * stream ,
                              char * name , char * type ,
                              unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                              unsigned char noColor ,
                              float * coords ,
                              unsigned int coordLength ,
                              float scaleX,
                              float scaleY,
                              float scaleZ,
                              unsigned int particleNumber
                            );
int removeObjectFromVirtualStream(struct VirtualStream * stream , unsigned int ObjID );

int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * type , char * model
                                );

int addEventToVirtualStream(
                             struct VirtualStream * stream ,
                             unsigned int objIDA ,
                             unsigned int objIDB ,
                             unsigned int eventType ,
                             char * data ,
                             unsigned int dataSize
                           );


void myStrCpy(char * destination,char * source,unsigned int maxDestinationSize);

#endif // TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
