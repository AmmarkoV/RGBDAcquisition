#ifndef MODEL_LOADER_H_INCLUDED
#define MODEL_LOADER_H_INCLUDED

enum ModelTypes
{
    NOTYPE = 0 ,
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

    float colorR;
    float colorG;
    float colorB;

    float transparency;
    int nocull;

};


struct Model * loadModel(char * modelname);
void unloadModel(struct Model * mod);

void drawModelAt(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
void drawModel(struct Model * mod);

int addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
int addToModelCoordinatesNoSTACK(struct Model * mod,float *x,float *y,float *z,float *heading,float *pitch,float *roll);

int setModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
int setModelCoordinatesNoSTACK(struct Model * mod,float * x,float* y,float *z,float *heading,float *pitch,float* roll);
int setModelColor(struct Model * mod,float *R,float *G,float *B,float *transparency);

int drawCube();

#endif // MODEL_LOADER_H_INCLUDED
