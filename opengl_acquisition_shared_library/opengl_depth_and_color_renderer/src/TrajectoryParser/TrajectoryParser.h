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


    unsigned int MAX_numberOfConnectors;
    unsigned int numberOfConnectors;
    struct VirtualConnector *connector;

    unsigned int MAX_numberOfEvents;
    unsigned int numberOfEvents;
    struct VirtualEvent * event;

    double scaleWorld[6];
    int rotationsOverride;
    int rotationsXYZ[3];
    float rotationsOffset[3];

    unsigned int userCanMoveCameraOnHisOwn;
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

/**
* @brief Get the ObjectID by using the name of an Object
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the object
* @param Output unsigned int that is set to 1 if we found the object , or 0 if we didn't find the object
* @retval ObjID if found is 1 can also be 0 ( ObjID 0 is the camera */
ObjectIDHandler getObjectID(struct VirtualStream * stream,char * name, unsigned int * found);

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


/**
* @brief Write a VirtualStream to a file  , so state can be loaded on another run/machine etc
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the output file that will hold the text file describing this particular virtual stream
* @bug The writeVirtualStream function hasn't been updated for a long time and probably no longer reflects the full state of a virtual stream
* @retval 1=Success , 0=Failure */
int writeVirtualStream(struct VirtualStream * newstream,char * filename);

/**
* @brief Read a VirtualStream from a file
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the input file that will populate the new virtual stream
* @bug  Code of readVirtualStream is *quickly* turning to shit after a chain of unplanned insertions on the parser
   This should probably be split down to some primitives and also support things like including a file from another file dynamic reload of models/objects explicit support for Quaternions / Rotation Matrices and getting rid of some intermediate
   parser declerations like arrowsX or objX
* @retval 1=Success , 0=Failure */
int readVirtualStream(struct VirtualStream * newstream/* , char * filename*/);


/**
* @brief Create and allocate the structures to accomodate a virtual stream , and read it from a file ( see readVirtualStream )
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Name of the input file that will populate the new virtual stream
* @retval 0=Failure , anything else is a pointer to a valid struct VirtualStream * stream */
struct VirtualStream * createVirtualStream(char * filename);

/**
* @brief Destroy and deallocate a virtual stream
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @retval 1=Success , 0=Failure */
int destroyVirtualStream(struct VirtualStream * stream);





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

/**
* @brief Remove an existing Object from a Virtual stream
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object ID we want to remove
* @bug This is a stub function , removeObjectFromVirtualStream needs to be properly implemented
* @retval 1=Success , 0=Failure */
int removeObjectFromVirtualStream(struct VirtualStream * stream ,  unsigned int ObjID );




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
                              char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength,
                               float scaleX , float scaleY ,float scaleZ ,
                              float R , float G , float B , float Alpha
                       );

/**
* @brief Create a new Object Type definition
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object Type we want to Create
* @param The model it will use
* @retval 1=Success , 0=Failure */
int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * type , char * model
                                );

/**
* @brief Calculate the position for an object at an absolute time interval
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object Id we want to get info about
* @param Time in milliseconds ( absolute time value in milliseconds )
* @param Output Array of floats , should be at least 4 floats long
* @retval 1=Success , 0=Failure */
int calculateVirtualStreamPos(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeMilliseconds,float * pos, float * scaleX , float * scaleY ,float * scaleZ);

/**
* @brief Calculate the position for an object after a delta time interval
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object Id we want to get info about
* @param Time in milliseconds ( a delta that has to be combined with last value , milliseconds )
* @param Output Array of floats , should be at least 4 floats long
* @retval 1=Success , 0=Failure */
int calculateVirtualStreamPosAfterTime(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAfterMilliseconds,float * pos, float * scaleX , float * scaleY ,float * scaleZ);


/**
* @brief Get an array of Floats , describing the last position of the objects
* @ingroup trajectoryParser
* @param Pointer to a valid stream
* @param Object Id we want to get info about
* @param Output Array of floats , should be at least 4 floats long
* @retval 1=Success , 0=Failure */
int getVirtualStreamLastPosF(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos, float * scaleX , float * scaleY ,float * scaleZ);


#ifdef __cplusplus
}
#endif

#endif // TRAJECTORYPARSER_H_INCLUDED
