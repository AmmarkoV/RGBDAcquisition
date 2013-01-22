#ifndef SCENE_H_INCLUDED
#define SCENE_H_INCLUDED

#define WIDTH 640
#define HEIGHT 480

extern float farPlane;
extern float nearPlane;

int renderScene();
int initScene();
int tickScene();
int closeScene();
#endif // VISUALS_H_INCLUDED
