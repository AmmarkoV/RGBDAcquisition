/** @file model_loader.h
 *  @brief  A module that loads models from files so that they can be rendered
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef MODEL_LOADER_H_INCLUDED
#define MODEL_LOADER_H_INCLUDED



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
    //-----------
    OBJMODEL,
    //-----------
    TOTAL_POSSIBLE_MODEL_TYPES
};


/**
* @brief The structure that defines what a Model Consists of
*/
struct Model
{
    //Pointer to the model read in memory ( what model_loader_obj reads )
    void * model;

    int type; //See enum ModelTypes

    //Position / Dimensions
    float x , y , z , heading , pitch , roll , scaleX , scaleY ,scaleZ;

    //Color
    float colorR , colorG , colorB , transparency;

    //Flags
    unsigned char nocull , nocolor;
};


#define MAX_MODEL_PATHS 120


/**
* @brief Load a model from a file
* @ingroup ModelLoader
* @param String of filename of the file to load
* @param The "friendly" name of the model loaded
* @retval 0=Error , A pointer to a model structure
*/
struct Model * loadModel(char * directory,char * modelname);


/**
* @brief Unload a loaded model
* @ingroup ModelLoader
* @param A loaded model
* @retval 0=Error , 1=Success
*/
void unloadModel(struct Model * mod);


int drawConnector(
                  float * posA,
                  float * posB,
                  float * scale ,
                  unsigned char R , unsigned char G , unsigned char B , unsigned char Alpha );

/**
* @brief Draw a Model at a specified pose in our world
* @ingroup ModelLoader
* @param A loaded model
* @param X coordinates of model
* @param Y coordinates of model
* @param Z coordinates of model
* @param heading of model
* @param pitch of model
* @param roll of model
* @retval 0=Error , 1=Success
*/
int drawModelAt(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);

/**
* @brief Draw a Model at current position
* @ingroup ModelLoader
* @param A loaded model
* @retval 0=Error , 1=Success
*/
int drawModel(struct Model * mod);


/**
* @brief Add to current coordinates of model
* @ingroup ModelLoader
* @param A loaded model
* @param X coordinates of model
* @param Y coordinates of model
* @param Z coordinates of model
* @param heading of model
* @param pitch of model
* @param roll of model
* @retval 0=Error , 1=Success
*/
int addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);

/**
* @brief Add to current coordinates of model by passing pointers and not the values
* @bug This function is provided because passing variables through the stack does not work ( Stack corruption ? )
* @ingroup ModelLoader
* @param A loaded model
* @param X coordinates of model
* @param Y coordinates of model
* @param Z coordinates of model
* @param heading of model
* @param pitch of model
* @param roll of model
* @retval 0=Error , 1=Success
*/
int addToModelCoordinatesNoSTACK(struct Model * mod,float *x,float *y,float *z,float *heading,float *pitch,float *roll);

/**
* @brief Set current coordinates of model
* @ingroup ModelLoader
* @param A loaded model
* @param X coordinates of model
* @param Y coordinates of model
* @param Z coordinates of model
* @param heading of model
* @param pitch of model
* @param roll of model
* @retval 0=Error , 1=Success
*/
int setModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);


/**
* @brief Set current coordinates of model by passing pointers and not the values
* @bug This function is provided because passing variables through the stack does not work ( Stack corruption ? )
* @ingroup ModelLoader
* @param A loaded model
* @param X coordinates of model
* @param Y coordinates of model
* @param Z coordinates of model
* @param heading of model
* @param pitch of model
* @param roll of model
* @retval 0=Error , 1=Success
*/
int setModelCoordinatesNoSTACK(struct Model * mod,float * x,float* y,float *z,float *heading,float *pitch,float* roll);


/**
* @brief Set current color of model
* @ingroup ModelLoader
* @bug   Transparency needs a correct drawing order
* @param A loaded model
* @param R Color channel
* @param G Color channel
* @param B Color channel
* @param Transaperncy color channel
* @param No color byte
* @retval 0=Error , 1=Success
*/
int setModelColor(struct Model * mod,float *R,float *G,float *B,float *transparency,unsigned char * noColor);


/**
* @brief Get Model bounding box
* @ingroup ModelLoader
* @param Output MinimumX
* @param Output MinimumY
* @param Output MinimumZ
* @param Output MaximumX
* @param Output MaximumY
* @param Output MaximumZ
* @retval 0=Error , 1=Success
*/
int getModelBBox(struct Model *mod , float * minX,  float * minY , float * minZ , float * maxX , float * maxY , float * maxZ);


/**
* @brief Get Model 3D Size
* @ingroup ModelLoader
* @param Output Size X
* @param Output Size Y
* @param Output Size Z
* @retval 0=Error , 1=Success
*/
int getModel3dSize(struct Model *mod , float * sizeX , float * sizeY , float * sizeZ );


/**
* @brief Draw a Cube
* @ingroup ModelLoader
* @retval 0=Error , 1=Success
*/
int drawCube();

#endif // MODEL_LOADER_H_INCLUDED
