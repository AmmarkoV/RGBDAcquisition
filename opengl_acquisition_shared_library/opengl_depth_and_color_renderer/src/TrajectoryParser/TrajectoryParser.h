/** @file TrajectoryParser.h
 *  @brief Parsing trajectories of 3d Objects to create 3d visualizations
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef TRAJECTORYPARSER_H_INCLUDED
#define TRAJECTORYPARSER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/**  @brief The maximum path for Object Files/Object Names/Object Types  etc */
#define MAX_PATH 250

#include "../ModelLoader/model_loader.h"
#include "hashmap.h"

#define LINE_MAX_LENGTH 1024
#define OBJECT_TYPES_TO_ADD_STEP 10
#define OBJECTS_TO_ADD_STEP 10
#define EVENTS_TO_ADD_STEP 10
#define FRAMES_TO_ADD_STEP 123

#define PRINT_DEBUGGING_INFO 0
#define PRINT_WARNING_INFO 0
#define PRINT_LOAD_INFO 0

#define CASE_SENSITIVE_OBJECT_NAMES 0

#define USE_QUATERNIONS_FOR_ORBITING 0

//-------------------------------------------------------

//This is retarded , i have to remake parsing to fix this
#define INCREMENT_TIMER_FOR_EACH_OBJ 0


extern float depthMemoryOutputScale;


/**
* @brief Some type definitions to better seperate ObjectID , TypeID concepts when used as function parameters
*/
typedef unsigned int ObjectIDHandler;
typedef unsigned int ObjectTypeID;


/**
* @brief Whats the state of our playback of the stream
*/
enum PlaybackState
{
    STOPPED = 0 ,
    PLAYING,
    PLAYING_BACKWARDS
};


struct Joint
{
  char altered;
  float x , y , z , scaleX ,scaleY, scaleZ;
  char useEulerRotation;
  char useQuaternion;
  float rot1 , rot2 , rot3 , rot4;
  char useMatrix4x4;
  float m[16];
};


struct JointState
{
 struct Joint * joint;
 unsigned int numberOfJoints;
};


/**
* @brief This holds the state for each known frame/position of each object!
*/
struct KeyFrame
{
   float x , y , z , scaleX ,scaleY, scaleZ;
   float rot1 , rot2 , rot3 , rot4;
   unsigned char isQuaternion;

   float R , G , B , Alpha ;
   unsigned char hasColor;
   unsigned char hasTrans;

   struct JointState * jointList;
   unsigned char hasNonDefaultJointList;

   //TimeStamp in milliseconds
   unsigned int time;
};


/**
* @brief ObjectTypes provide a lookup table to associate user-given abstract types to existing 3D models
*        models can be paths to 3d compatible object files ( .obj ) or hardcoded objects like planes , grids etc (, see model_loader.c)
*/
struct ObjectType
{
   char name[MAX_PATH+1];
   char model[MAX_PATH+1];
   unsigned int modelListArrayNumber;
   unsigned int numberOfBones;
};



/**
* @brief Handled Events list
*/
enum EventType
{
  EVENT_EMPTY = 0 ,
  EVENT_INTERSECTION,
  //--------------------
  NUMBER_OF_EVENTS
};



/**
* @brief VirtualEvent structure holds all the data that defines events
*/
struct VirtualEvent
{
    unsigned int eventType;
    unsigned int objID_A;
    unsigned int objID_B;

    char * data;
    unsigned int dataSize;

    unsigned char activated;
};



/**
* @brief VirtualConnector structure holds all the data that defines a connector between two objects
*/
struct VirtualConnector
{
   char firstObject[MAX_PATH+1];
   char secondObject[MAX_PATH+1];
   char typeStr[MAX_PATH+1];

   unsigned int connectorType;
   unsigned int objID_A;
   unsigned int objID_B;

   float R , G , B , Transparency;

   double scale;

};







/**
* @brief VirtualObject structure holds all the data that defines an object
*/
struct VirtualObject
{
   char name[MAX_PATH+1];
   char typeStr[MAX_PATH+1];
   char value[MAX_PATH+1];
   ObjectTypeID type;

   float R , G , B , Transparency;
   unsigned char nocolor;

   unsigned int MAX_timeOfFrames;
   unsigned int MAX_numberOfFrames;
   unsigned int numberOfFrames;
   struct KeyFrame * frame;


   unsigned int lastCalculationTime;

   unsigned int lastFrame;
   unsigned int lastFrameTime;
   unsigned int nextFrameTime;

   double scaleX,scaleY,scaleZ;
   unsigned int particleNumber;

   unsigned int bbox2D[4];


   unsigned char hasAssociatedEvents;


   /*
   // 0-6  low bounds          X Y Z    A B C D
   // 7-13 high bounds         X Y Z    A B C D
   // 14-20 resample variances X Y Z    A B C D
   double limits[ 7 * 3];

   unsigned int generations;
   unsigned int particles;*/
};



/**
* @brief A VirtualObject structure holds many objects each of which have their own characteristics
*/
struct VirtualStream
{

    //--------------------------------------------------------
    // These are the matrices actively used by OpenGL
    // to render the scene
    //--------------------------------------------------------
     double activeProjectionMatrix[16];
     double activeModelViewMatrix[16];
     double activeModelViewProjectionMatrix[16];
     double activeNormalTransformation[16];
    //--------------------------------------------------------
    //--------------------------------------------------------


    //--------------------------------------------------------
    // These are matrices declared on the scene file that
    // control the view of the camera for our scene
    //--------------------------------------------------------
    int projectionMatrixDeclared;
    double projectionMatrix[16];

    int modelViewMatrixDeclared;
    double modelViewMatrix[16];


    int emulateProjectionMatrixDeclared;
    double emulateProjectionMatrix[9];

    int extrinsicsDeclared;
    double extrinsicTranslation[3];
    double extrinsicRodriguezRotation[3];
    //--------------------------------------------------------
    //--------------------------------------------------------
    //--------------------------------------------------------

    float backgroundR,backgroundG,backgroundB;

    unsigned int MAX_numberOfObjectTypes;
    unsigned int numberOfObjectTypes;
    struct ObjectType * objectTypes;
    struct hashMap * objectTypesHash;


    unsigned int MAX_numberOfObjects;
    unsigned int numberOfObjects;
    unsigned int selectedObject;
    struct VirtualObject * object;
    struct hashMap * objectHash;


    unsigned int MAX_numberOfConnectors;
    unsigned int numberOfConnectors;
    struct VirtualConnector *connector;
    struct hashMap * connectorHash;

    unsigned int MAX_numberOfEvents;
    unsigned int numberOfEvents;
    struct VirtualEvent * event;
    struct hashMap * eventHash;

    struct ModelList * associatedModelList;

    double scaleWorld[6];
    int rotationsOverride;
    int rotationsXYZ[3];
    float rotationsOffset[3];

    unsigned int userCanMoveCameraOnHisOwn;
    unsigned int playback;
    float rate;

    unsigned int autoRefresh;
    unsigned int autoRefreshForce;
    unsigned int lastRefresh;
    unsigned int fileSize;

    char showSkeleton;
    char alwaysShowLastFrame;
    char ignoreTime;
    char reverseLoop;
    char debug;
    char silent;


    char renderWireframe;

    unsigned int objDeclarationsOffset;
    unsigned int timestamp;
    unsigned int ticks;

    char filename[MAX_PATH+1];
};

/**
* @brief Get the ObjectID by using the name of an Object
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the object
* @param Output unsigned int that is set to 1 if we found the object , or 0 if we didn't find the object
* @retval ObjID if found is 1 can also be 0 ( ObjID 0 is the camera */
ObjectIDHandler getObjectID(struct VirtualStream * stream,const char * name, unsigned int * found);

/**
* @brief Get a String with the model of a typeID
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param  ObjectTypeID
* @retval String with the model associated with this typeID */
char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID);

/**
* @brief Get the Color/Transparency State for a specific Object
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param ObjID that defines the object to use
* @param Output R channel color
* @param Output G channel color
* @param Output B channel color
* @param Output Alpha(Transparency) channel color
* @param Output NoColor flag ( 1=Set 0=Not Set )
* @retval 1=Success,0=Failure */
int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency, unsigned char * noColor);




/**
* @brief Get a String with the model from an ObjID
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param  ObjID
* @retval String with the model associated with this ObjID */
char * getModelOfObjectID(struct VirtualStream * stream,ObjectIDHandler ObjID);




int objectsCollide(struct VirtualStream * newstream,unsigned int atTime,unsigned int objIDA,unsigned int objIDB);


int appendVirtualStreamFromFile(struct VirtualStream * newstream , struct ModelList * modelStorage,const char * filename);

/**
* @brief Write a VirtualStream to a file  , so state can be loaded on another run/machine etc
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the output file that will hold the text file describing this particular virtual stream
* @bug The writeVirtualStream function hasn't been updated for a long time and probably no longer reflects the full state of a virtual stream
* @retval 1=Success , 0=Failure */
int writeVirtualStream(struct VirtualStream * newstream,const char * filename);

/**
* @brief Read a VirtualStream from a file
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Pointer to a model list that will accomodate the newly loaded models of this virtual stream
* @bug  Code of readVirtualStream is *quickly* turning to shit after a chain of unplanned insertions on the parser
   This should probably be split down to some primitives and also support things like including a file from another file dynamic reload of models/objects explicit support for Quaternions / Rotation Matrices and getting rid of some intermediate
   parser declerations like arrowsX or objX
* @retval 1=Success , 0=Failure */
int readVirtualStream(struct VirtualStream * newstream,struct ModelList * modelStorage);


/**
* @brief Create and allocate the structures to accomodate a virtual stream , and read it from a file ( see readVirtualStream )
* @ingroup trajectoryParser
* @param Name of the input file that will populate the new virtual stream
* @param Pointer to a model list that will accomodate the newly loaded models of this virtual stream
* @retval 0=Failure , anything else is a pointer to a valid struct VirtualStream * stream */
struct VirtualStream * createVirtualStream(const char * filename,struct ModelList * modelStorage);

/**
* @brief Destroy and deallocate a virtual stream
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @retval 1=Success , 0=Failure */
int destroyVirtualStream(struct VirtualStream * stream);




int refreshVirtualStream(struct VirtualStream * newstream,struct ModelList * modelStorage);


/**
* @brief Remove an existing Object from a Virtual stream
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object ID we want to remove
* @bug This is a stub function , removeObjectFromVirtualStream needs to be properly implemented
* @retval 1=Success , 0=Failure */
int removeObjectFromVirtualStream(struct VirtualStream * stream ,  unsigned int ObjID );




#ifdef __cplusplus
}
#endif

#endif // TRAJECTORYPARSER_H_INCLUDED
