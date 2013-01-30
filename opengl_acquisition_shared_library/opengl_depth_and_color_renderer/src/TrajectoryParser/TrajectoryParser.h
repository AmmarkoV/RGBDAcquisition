#ifndef TRAJECTORYPARSER_H_INCLUDED
#define TRAJECTORYPARSER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int ObjectIDHandler;
typedef unsigned int ObjectTypeID;

enum PlaybackState
{
    STOPPED = 0 ,
    PLAYING,
    PLAYING_BACKWARDS
};

struct KeyFrame
{
   float x; float y; float z;
   float rot1; float rot2; float rot3; float rot4;
   unsigned int time;
};


struct ObjectType
{
   char name[15];
   char model[15];
};


struct VirtualObject
{
   char name[15];
   char typeStr[15];
   char value[15];
   ObjectTypeID type;

   float R;
   float G;
   float B;
   float Transparency;


   unsigned int MAX_timeOfFrames;
   unsigned int MAX_numberOfFrames;
   unsigned int numberOfFrames;
   struct KeyFrame * frame;


   unsigned int lastCalculationTime;

   unsigned int lastFrame;
   unsigned int lastFrameTime;
   unsigned int nextFrameTime;

};

struct VirtualStream
{
    unsigned int MAX_numberOfObjectTypes;
    unsigned int numberOfObjectTypes;
    struct ObjectType * objectTypes;



    unsigned int MAX_numberOfObjects;
    unsigned int numberOfObjects;
    struct VirtualObject * object;

    unsigned int playback;

    unsigned int autoRefresh;
    unsigned int lastRefresh;
    unsigned int fileSize;

    char ignoreTime;
    char reverseLoop;

    char filename[250];
};

ObjectIDHandler getObjectID(struct VirtualStream * stream,char * name, unsigned int * found);
char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID);
int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency);

int readVirtualStream(struct VirtualStream * newstream , char * filename);
struct VirtualStream * createVirtualStream(char * filename);
int destroyVirtualStream(struct VirtualStream * stream);

int calculateVirtualStreamPos(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeMilliseconds,float * pos);
int calculateVirtualStreamPosAfterTime(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAfterMilliseconds,float * pos);




#ifdef __cplusplus
}
#endif

#endif // TRAJECTORYPARSER_H_INCLUDED
