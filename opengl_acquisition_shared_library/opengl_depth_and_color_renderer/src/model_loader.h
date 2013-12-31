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
    OBJMODEL
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
    float x , y , z , heading , pitch , roll , scale;

    //Color
    float colorR , colorG , colorB , transparency;

    //Flags
    unsigned char nocull , nocolor;
};


#define MAX_MODEL_PATHS 120

struct Model * loadModel(char * directory,char * modelname);
void unloadModel(struct Model * mod);

int drawModelAt(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
int drawModel(struct Model * mod);

int addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
int addToModelCoordinatesNoSTACK(struct Model * mod,float *x,float *y,float *z,float *heading,float *pitch,float *roll);

int setModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
int setModelCoordinatesNoSTACK(struct Model * mod,float * x,float* y,float *z,float *heading,float *pitch,float* roll);
int setModelColor(struct Model * mod,float *R,float *G,float *B,float *transparency,unsigned char * noColor);

int getModelBBox(struct Model *mod , float * minX,  float * minY , float * minZ , float * maxX , float * maxY , float * maxZ);
int getModel3dSize(struct Model *mod , float * sizeX , float * sizeY , float * sizeZ );

int drawCube();

#endif // MODEL_LOADER_H_INCLUDED
