/** @file model_loader.h
 *  @brief  A module that loads models from files so that they can be rendered
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef MODEL_LOADER_H_INCLUDED
#define MODEL_LOADER_H_INCLUDED


#define MAX_MODEL_PATHS 250

/**
* @brief The structure that defines what a Model Consists of
*/
struct Model
{
    //Pointer to the model read in memory ( what model_loader_obj reads )
    void * modelInternalData;
    //This can actually be different types of struct depending on if we are talking about an obj/tri/hardcoded/etc model
    //We know the type byaccessing the "type" attribute of this struct

    int type; //See enum ModelTypes
    unsigned int numberOfBones;


    float bbox2D[4]; //The 2D rendering output bounding box of the specific model

    float x , y , z , heading , pitch , roll , scaleX , scaleY ,scaleZ;
    float minX,minY,minZ,maxX,maxY,maxZ;

    //Color
    float colorR , colorG , colorB , transparency;

    //Flags
    unsigned char nocull , nocolor , wireframe , highlight;

    //-----------------
    char pathOfModel[MAX_MODEL_PATHS];
};



struct ModelList
{
  struct Model * models;
  unsigned int currentNumberOfModels;
  unsigned int MAXNumberOfModels;
};

struct ModelList *  allocateModelList(unsigned int initialSpace);


int printModelList(struct ModelList* modelStorage);

int loadModelToModelList(struct ModelList* modelStorage,char * modelDirectory,char * modelName , unsigned int * whereModelWasLoaded);

/**
* @brief Update Model Position triggers , ( 3D / 2D )
* @ingroup ModelLoader
* @param Pointer to a model object
* @param Pointer to a position array ( X,Y,Z and Quat or Euler )
* @retval 1=Success,0=Failure
* @bug If updateModelPosition is called without the correct modelview/projection matrices loaded it will fail miserably , it should probably not even be exposed outside the modelloader.c file..
*/
unsigned int updateModelPosition(struct Model * model,float * position);


/**
* @brief Find an already loaded model
* @ingroup ModelLoader
* @param Pointer of model array
* @param String of filename of the file to load
* @param The "friendly" name of the model loaded
* @param Did the operation find something..?
* @retval 0=Could not find model , A pointer to an already loaded model structure
*/
//unsigned int findModel(struct ModelList * modelStorage , char * directory,char * modelname ,int * found );



/**
* @brief Load a model from a file
* @ingroup ModelLoader
* @param Pointer to the model list
* @param Where to write the loaded model
* @param String of filename of the file to load
* @param The "friendly" name of the model loaded
* @retval 0=Error , A pointer to a model structure
*/
//unsigned int loadModel(struct ModelList* modelStorage , unsigned int whereToLoadModel , char * directory,char * modelname , unsigned int * loadedModelNumber);


/**
* @brief Unload a loaded model
* @ingroup ModelLoader
* @param A loaded model
* @retval 0=Error , 1=Success
*/
void unloadModel(struct Model * mod);


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


int getModelListBoneNumber(struct ModelList * modelStorage,unsigned int modelNumber);

//TODO Add explanation here
int getModelBoneIDFromBoneName(struct Model *mod,char * boneName,int * found);

#endif // MODEL_LOADER_H_INCLUDED
