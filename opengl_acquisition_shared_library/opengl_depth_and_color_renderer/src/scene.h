#ifndef SCENE_H_INCLUDED
#define SCENE_H_INCLUDED

extern int WIDTH;
extern int HEIGHT;

extern float farPlane;
extern float nearPlane;

extern int useIntrinsicMatrix;
extern double cameraMatrix[9];


extern int useCustomModelViewMatrix;
extern double customModelViewMatrix[16];
extern double customTranslation[3];
extern double customRodriguezRotation[3];

int renderScene();
int renderPhotoshoot(int objID, float angleX,float angleY,float angleZ);
int initScene(char * confFile);
int tickScene();
int closeScene();
#endif // VISUALS_H_INCLUDED
