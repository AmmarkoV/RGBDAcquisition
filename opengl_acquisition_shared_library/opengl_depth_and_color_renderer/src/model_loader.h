#ifndef MODEL_LOADER_H_INCLUDED
#define MODEL_LOADER_H_INCLUDED

enum ModelTypes
{
    NOTYPE = 0 ,
    OBJ_AXIS,
    OBJ_PLANE,
    OBJ_CUBE,
    OBJMODEL
};


struct Model
{
    void * model;
    int type;
    //other stuff here
    float x;
    float y;
    float z;

    float heading;
    float pitch;
    float roll;

    float scale;

    float colorR;
    float colorG;
    float colorB;

    float transparency;
    unsigned char nocull;
    unsigned char nocolor;

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

int drawCube();

#endif // MODEL_LOADER_H_INCLUDED
