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


int setupPhotoshoot(
                     void * context,
                     int objID,
                     unsigned int columns , unsigned int rows ,
                     float distance,
                     float angleX,float angleY,float angleZ ,
                     float angXVariance ,float angYVariance , float angZVariance
                   );

void * createPhotoshoot(
                        int objID,
                        unsigned int columns , unsigned int rows ,
                        float distance,
                        float angleX,float angleY,float angleZ ,
                        float angXVariance ,float angYVariance , float angZVariance
                       );

int renderPhotoshoot( void * context );



int initScene(char * confFile);
int tickScene();
int closeScene();
#endif // VISUALS_H_INCLUDED
