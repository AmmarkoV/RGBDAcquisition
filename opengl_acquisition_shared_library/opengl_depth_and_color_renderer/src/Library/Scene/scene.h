/** @file scene.h
 *  @brief  Basic scene information
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef SCENE_H_INCLUDED
#define SCENE_H_INCLUDED


#include "../TrajectoryParser/TrajectoryParser.h"

extern unsigned int WIDTH;
extern unsigned int HEIGHT;

struct VirtualStream *  getLoadedScene();
struct ModelList *  getLoadedModelStorage();



float sceneGetNearPlane();

int sceneSetNearFarPlanes(float near, float far);
float sceneGetDepthScalingPrameter();

int sceneSeekTime(unsigned int seekTime);
int sceneIgnoreTime(unsigned int newSettingMode);


int sceneSetOpenGLExtrinsicCalibration(struct VirtualStream * scene,float * rodriguez,float * translation ,float scaleToDepthUnit);

int sceneSetOpenGLIntrinsicCalibration(struct VirtualStream * scene,float * camera);



int sceneSetOpenGLIntrinsicCalibrationNew(struct VirtualStream * scene,float fx,float fy,float cx,float cy,float width,float height,float nearPlane,float farPlane);

int setupSceneCameraBeforeRendering(struct VirtualStream * scene);

/**
* @brief Render A Scene
* @ingroup Scene
* @retval 1=Success , 0=Failure
*/
int renderScene();

/**
* @brief Update window size
* @ingroup Scene
* @param New Width
* @param New Height
* @retval 1=Success , 0=Failure
*/
int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight);

int handleUserInput(char key,int state,unsigned int x, unsigned int y);

unsigned int *  getObject2DBoundingBoxList(unsigned int * bboxItemsSize);
int updateProjectionMatrix();


/**
* @brief Render A Scene in "photoshoot" mode , means take multiple "photos" according to a createPhotoshoot context
* @ingroup Scene
* @param A context created with void * createPhotoshoot
* @retval 1=Success , 0=Failure
*/
int renderPhotoshoot( void * context );



/**
* @brief Initialize a scene using a configuration file
* @ingroup Scene
* @param argc ( can be null )
* @param argv ( can be null )
* @param String with a path to a configuration file
* @retval 1=Success , 0=Failure
*/
int initScene(int argc,const char *argv[],const char * confFile);

/**
* @brief Tick time so that model positions in the scene are updated
* @ingroup Scene
* @retval 1=Success , 0=Failure
*/
int tickScene(unsigned int framerate);


/**
* @brief Stop Using the Scene
* @ingroup Scene
* @retval 1=Success , 0=Failure
*/
int closeScene();



/**
* @brief Set Keyboard Control
* @ingroup Scene
* @param New State
* @retval 1=Success , 0=Failure
*/
int sceneSwitchKeyboardControl(int newVal);
#endif // VISUALS_H_INCLUDED
