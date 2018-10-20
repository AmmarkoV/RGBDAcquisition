#ifndef TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
#define TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED


#include "TrajectoryParser.h"

#ifdef __cplusplus
extern "C" {
#endif


int growVirtualStreamFrames(struct VirtualObject * streamObj,unsigned int framesToAdd);
int growVirtualStreamObjectsTypes(struct VirtualStream * stream,unsigned int objectsTypesToAdd);
int growVirtualStreamObjects(struct VirtualStream * stream,unsigned int objectsToAdd);
int growVirtualStreamEvents(struct VirtualStream * stream,unsigned int eventsToAdd);
int growVirtualStreamConnectors(struct VirtualStream * stream,unsigned int connectorsToAdd);




int dummy_strcasecmp_internal(char * input1, char * input2);

void listAllObjectTypeID(struct VirtualStream * stream);

ObjectIDHandler getObjectID(struct VirtualStream * stream,const char * name, unsigned int * found);


ObjectTypeID getObjectTypeID(struct VirtualStream * stream,const char * typeName,unsigned int * found);

char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID);

char * getModelOfObjectID(struct VirtualStream * stream,ObjectIDHandler ObjID);

int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency, unsigned char * noColor);
unsigned long getFileSize(char * filename);




/**
* @brief Add a new Position at a specified time for an object using its ObjectID
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object ID we want to add the position to
* @param The time ine Milliseconds
* @retval 1=Success , 0=Failure */
int addStateToObjectID(
                               struct VirtualStream * stream ,
                               unsigned int ObjID  ,
                               unsigned int timeMilliseconds ,
                               float * coord ,
                               unsigned int coordLength ,
                               float scaleX , float scaleY ,float scaleZ ,
                               float R , float G , float B , float Alpha
                       );



/**
* @brief Add a new Position at a specified time for an object using its name
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param String with the name of the object we want to add the position to
* @param The time ine Milliseconds
* @retval 1=Success , 0=Failure */
int addStateToObject(
                              struct VirtualStream * stream ,
                              const char * name  ,
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


int addPoseToObjectState(
                              struct VirtualStream * stream ,
                              struct ModelList * modelStorage,
                              char * name  ,
                              char * jointName,
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

int generateAngleObjectsForVirtualStream(struct VirtualStream * stream, struct ModelList * modelStorage,char * excludeObjectType);


/**
* @brief Add a new Object to the Virtual stream
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the new object
* @param Type of the new object ( see ObjectTypes )
* @param R channel color
* @param G channel color
* @param B channel color
* @param Alpha(Transparency) channel color
* @param NoColor flag ( 1=Set 0=Not Set )
* @param Pointer to an array of floats that defines initial position ( should be 3 or 4 floats long )
* @param The length of the coords array ( typically 3 or 4 )
* @param Particle number
* @retval 1=Success , 0=Failure */
int addObjectToVirtualStream(
                              struct VirtualStream * stream ,
                              struct ModelList * modelStorage,
                              const char * name ,const char * type ,
                              unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                              unsigned char noColor ,
                              float * coords ,
                              unsigned int coordLength ,
                              float scaleX,
                              float scaleY,
                              float scaleZ,
                              unsigned int particleNumber
                            );

int addSimpleObject(
                    struct VirtualStream * stream ,
                    const char * type ,
                    unsigned char R, unsigned char G , unsigned char B ,
                    float * coords ,
                    float scale
                   );

int removeObjectFromVirtualStream(struct VirtualStream * stream , unsigned int ObjID );

int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 const char * type , const char * model,
                                 const char * webLink
                                );

int addEventToVirtualStream(
                             struct VirtualStream * stream ,
                             unsigned int objIDA ,
                             unsigned int objIDB ,
                             unsigned int eventType ,
                             char * data ,
                             unsigned int dataSize
                           );


void myStrCpy(char * destination,const char * source,unsigned int maxDestinationSize);


#ifdef __cplusplus
}
#endif

#endif // TRAJECTORYPARSERDATASTRUCTURES_H_INCLUDED
