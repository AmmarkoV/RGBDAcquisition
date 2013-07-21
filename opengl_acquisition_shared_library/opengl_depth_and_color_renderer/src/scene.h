#ifndef SCENE_H_INCLUDED
#define SCENE_H_INCLUDED

extern int WIDTH;
extern int HEIGHT;

extern float farPlane;
extern float nearPlane;

extern int useIntrinsicMatrix;
extern double cameraMatrix[9];


extern int useCustomMatrix;
extern float customMatrix[16];
extern float customTranslation[3];
extern float customRotation[3];

int renderScene();
int initScene();
int tickScene();
int closeScene();
#endif // VISUALS_H_INCLUDED
