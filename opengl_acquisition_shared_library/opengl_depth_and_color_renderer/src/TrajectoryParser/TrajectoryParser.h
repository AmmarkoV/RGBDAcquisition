#ifndef TRAJECTORYPARSER_H_INCLUDED
#define TRAJECTORYPARSER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_PATH 250

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
   char name[MAX_PATH+1];
   char model[MAX_PATH+1];
};


struct VirtualObject
{
   char name[MAX_PATH+1];
   char typeStr[MAX_PATH+1];
   char value[MAX_PATH+1];
   ObjectTypeID type;

   float R;
   float G;
   float B;
   float Transparency;
   unsigned char nocolor;

   unsigned int MAX_timeOfFrames;
   unsigned int MAX_numberOfFrames;
   unsigned int numberOfFrames;
   struct KeyFrame * frame;


   unsigned int lastCalculationTime;

   unsigned int lastFrame;
   unsigned int lastFrameTime;
   unsigned int nextFrameTime;

   double scale;


   // 0-6  low bounds          X Y Z    A B C D
   // 7-13 high bounds         X Y Z    A B C D
   // 14-20 resample variances X Y Z    A B C D
   double limits[ 7 * 3];

   int generations;
   int particles;

};

struct VirtualStream
{
    int projectionMatrixDeclared;
    double projectionMatrix[16];

    int modelViewMatrixDeclared;
    double modelViewMatrix[16];


    int emulateProjectionMatrixDeclared;
    double emulateProjectionMatrix[9];

    int extrinsicsDeclared;
    double extrinsicTranslation[3];
    double extrinsicRodriguezRotation[3];


    float backgroundR,backgroundG,backgroundB;

    unsigned int MAX_numberOfObjectTypes;
    unsigned int numberOfObjectTypes;
    struct ObjectType * objectTypes;

    unsigned int MAX_numberOfObjects;
    unsigned int numberOfObjects;
    struct VirtualObject * object;

    double scaleWorld[6];
    int rotationsOverride;
    int rotationsXYZ[3];
    float rotationsOffset[3];

    unsigned int playback;

    unsigned int autoRefresh;
    unsigned int lastRefresh;
    unsigned int fileSize;

    char ignoreTime;
    char reverseLoop;
    char debug;

    unsigned int timestamp;

    char filename[MAX_PATH+1];
};

ObjectIDHandler getObjectID(struct VirtualStream * stream,char * name, unsigned int * found);
char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID);
int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency, unsigned char * noColor);
char * getModelOfObjectID(struct VirtualStream * stream,ObjectIDHandler id);

int writeVirtualStream(struct VirtualStream * newstream,char * filename);
int readVirtualStream(struct VirtualStream * newstream , char * filename);
struct VirtualStream * createVirtualStream(char * filename);
int destroyVirtualStream(struct VirtualStream * stream);

int addObjectToVirtualStream(
                              struct VirtualStream * stream ,
                              char * name , char * type ,
                              unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                              unsigned char noColor ,
                              float * coords ,
                              unsigned int coordLength ,
                              float scale
                            );

int addPositionToObject(
                              struct VirtualStream * stream ,
                              char * name  ,
                              unsigned int time ,
                              float * coord ,
                              unsigned int coordLength
                       );

int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * type , char * model
                                );


int calculateVirtualStreamPos(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeMilliseconds,float * pos);
int calculateVirtualStreamPosAfterTime(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAfterMilliseconds,float * pos);
int getVirtualStreamLastPosF(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos);
int getVirtualStreamLastPosD(struct VirtualStream * stream,ObjectIDHandler ObjID,double * pos);


#ifdef __cplusplus
}
#endif

#endif // TRAJECTORYPARSER_H_INCLUDED
