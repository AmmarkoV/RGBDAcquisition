/** @file model_loader_hardcoded.h
 *  @brief  A module that knows how to draw models that are hardcoded inside here
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MODEL_LOADER_HARDCODED_H_INCLUDED
#define MODEL_LOADER_HARDCODED_H_INCLUDED

/**
* @brief An enumerator structure to id special Objects that are hardcoded in this library and don't need to be loaded using the OBJ loader
*/
enum ModelTypes
{
    NOTYPE = 0 ,
    OBJ_AXIS,
    OBJ_PLANE,
    OBJ_GRIDPLANE,
    OBJ_CUBE,
    OBJ_PYRAMID,
    OBJ_SPHERE,
    OBJ_INVISIBLE,
    OBJ_QUESTION,
    OBJ_BBOX,
    //-----------
    OBJ_MODEL,
    TRI_MODEL,
    OBJ_ASSIMP_MODEL,
    //-----------
    TOTAL_POSSIBLE_MODEL_TYPES
};



int initializeHardcodedCallLists();

/**
* @brief Check if model name is a hardcoded model
* @ingroup ModelLoaderHardcoded
* @param String of filename of the file to load
* @param Output that returns true if we have a hardcoded model name
* @retval 0=Enumerator of Hardcoded model
*/
unsigned int isModelnameAHardcodedModel(const char * modelname,unsigned int * itIsAHardcodedModel);



/**
* @brief Draw a hardcoded model described by modelType ( see enum ModelTypes ) at 0,0,0
* @ingroup ModelLoaderHardcoded
* @param see enum ModelTypes
* @retval 1=Success/0=Failure
*/
unsigned int drawHardcodedModelRaw(unsigned int modelType);



/**
* @brief Draw a hardcoded model described by modelType ( see enum ModelTypes ) at 0,0,0 ,
  This call will try to use a glCallList if it is available which greatly speeds up drawing
* @ingroup ModelLoaderHardcoded
* @param see enum ModelTypes
* @retval 1=Success/0=Failure
*/
unsigned int drawHardcodedModel(unsigned int modelType);


/**
* @brief Connectors are special hardcoded shapes that connect 2 points
* @ingroup ModelLoaderHardcoded
* @param Array for position of position A
* @param Array for position of position B
* @param Array for scale X,Y,Z
* @param R color [0-255]
* @param G color [0-255]
* @param B color [0-255]
* @param Alpha transparency [0-255]
* @retval 1=Success/0=Failure
*/
int drawConnector(float * posA,float * posB,float * scale ,
                  float * R , float * G , float * B , float * Alpha );






/**
* @brief Draw a bounding box providing the center and dimensions
* @ingroup ModelLoaderHardcoded
* @param Center Position X
* @param Center Position Y
* @param Center Position Z
* @param Minimum X
* @param Minimum Y
* @param Minimum Z
* @param Maximum X
* @param Maximum Y
* @param Maximum Z
* @retval 1=Success/0=Failure
*/
int drawBoundingBox(float x,float y,float z ,float minX,float minY,float minZ,float maxX,float maxY,float maxZ);













#endif // MODEL_LOADER_H_INCLUDED
