/** @file OGLRendererSandbox.h
 *  @brief  Interface to the OGL Renderer Sandbox
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef OGLRENDERERSANDBOX_H_INCLUDED
#define OGLRENDERERSANDBOX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

int doTest();


/**
* @brief Set OpenGL Near And Far Planes
* @ingroup OGLRendererSandbox
* @param Near plane
* @param Far plane
* @retval 0=Failure,1=Success
*/
int setOpenGLNearFarPlanes(double near , double far);


/**
* @brief Set OpenGL Intrinsic Parameters
* @ingroup OGLRendererSandbox
* @param an OpenGL Camera Matrix
* @bug Careful about providing Column/Row Major matrices
* @retval 0=Failure,1=Success
*/
int setOpenGLIntrinsicCalibration(double * camera);


/**
* @brief Set OpenGL Intrinsic Parameters
* @ingroup OGLRendererSandbox
* @param Rodriguez rotation vector ( 3 numbers opencv style )
* @param Translation vector ( 3 numbers )
* @param Scale of translation
* @retval 0=Failure,1=Success
*/
int setOpenGLExtrinsicCalibration(double * rodriguez,double * translation , double scaleToDepthUnit);



/**
* @brief Get 2D Width Of OpenGL Rendering Surface
* @ingroup OGLRendererSandbox
* @retval Width of Rendering Surface
*/
unsigned int getOpenGLWidth();

/**
* @brief Get 2D Height Of OpenGL Rendering Surface
* @ingroup OGLRendererSandbox
* @retval Height of Rendering Surface
*/
unsigned int getOpenGLHeight();

/**
* @brief Enable Keyboard control ( this has to be called after initializing a scene , otherwise it does nothing )
* @ingroup OGLRendererSandbox
* @param New Value of Keyboard setting
* @retval 0=Failure,1=Success
*/
int setKeyboardControl(int val);


/**
* @brief Enable Shaders ( this has to be called before initializing a scene , otherwise it does nothing )
* @ingroup OGLRendererSandbox
* @param Vertex Shader Filename
* @param Fragment Shader Filename
* @retval 0=Failure,1=Success
*/
int enableShaders(char * vertShaderFilename , char * fragShaderFilename );


/**
* @brief Get Virtual Focal Length
* @ingroup OGLRendererSandbox
* @retval Focal Length
*/
double getOpenGLFocalLength();

/**
* @brief Get Virtual Pixel Size
* @ingroup OGLRendererSandbox
* @retval Pixel Size
*/
double getOpenGLPixelSize();


/**
* @brief Get Z Buffer of Renderered Scene
* @ingroup OGLRendererSandbox
* @param The frame to fill with the Z Buffer
* @param Starting Point X
* @param Starting Point Y
* @param Width of depth to return
* @param Height of depth to return
* @retval 0=Failure,1=Success
*/
int getOpenGLZBuffer(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

/**
* @brief Get Depths of Renderered Scene
* @ingroup OGLRendererSandbox
* @param The frame to fill with the Depth
* @param Starting Point X
* @param Starting Point Y
* @param Width of depth to return
* @param Height of depth to return
* @retval 0=Failure,1=Success
*/
int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

/**
* @brief Get Colors of Renderered Scene
* @ingroup OGLRendererSandbox
* @param The frame to fill with the Color
* @param Starting Point X
* @param Starting Point Y
* @param Width of depth to return
* @param Height of depth to return
* @retval 0=Failure,1=Success
*/
int getOpenGLColor(char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height);


/**
* @brief Save Colors of Renderered Scene to File
* @ingroup OGLRendererSandbox
* @param Filename to write the color to
* @param Starting Point X
* @param Starting Point Y
* @param Width of depth to return
* @param Height of depth to return
* @retval 0=Failure,1=Success
*/
void writeOpenGLColor(char * colorfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);


/**
* @brief Save Depths of Renderered Scene to File
* @ingroup OGLRendererSandbox
* @param Filename to write the depth to
* @param Starting Point X
* @param Starting Point Y
* @param Width of depth to return
* @param Height of depth to return
* @retval 0=Failure,1=Success
*/
void writeOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);


/**
* @brief Start OGLRenderer Sandbox
* @ingroup OGLRendererSandbox
* @param Width of rendering surface
* @param Height of rendering surface
* @param Switch ( 1 = View Window , 0 = Disable Window )
* @param Filename of the scene file that describes the scene to be played out
* @retval 0=Failure,1=Success
*/
int startOGLRendererSandbox(unsigned int width,unsigned int height , unsigned int viewWindow ,char * sceneFile);



/**
* @brief Snap OGL Renderer Frame
* @ingroup OGLRendererSandbox
* @retval 0=Failure , 1=Success
*/
int snapOGLRendererSandbox();



/**
* @brief Stop OGL Renderer Sandbox
* @ingroup OGLRendererSandbox
* @retval 0=Failure , 1=Success
*/
int stopOGLRendererSandbox();

/* ---------------------------------------------------------
   ----------------- Photo shoot mode ----------------------
   --------------------------------------------------------- */
/**
* @brief Create a photoshoot context to later render it
* @ingroup OGLPhotoshoot
* @bug createOGLRendererPhotoshootSandbox call is providing the same arguments that snapOGLRendererPhotoshootSandbox uses , is there any point in it ?
* @param ObjectId to be drawn ( assumes that a scene has already been loaded )
* @param ObjectId to be drawn ( assumes that a scene has already been loaded )
* @param Number Of columns for the photoshoot
* @param Number Of rows for the photoshoot
* @param Central Rotation Angle on the X axis of the Object
* @param Central Rotation Angle on the Y axis of the Object
* @param Central Rotation Angle on the Z axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the X axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the Y axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the Z axis of the Object
* @retval 0=Failure, or a pointer to a photoshoot context
*/
void * createOGLRendererPhotoshootSandbox(
                                           int objID, unsigned int columns , unsigned int rows , float distance,
                                           float angleX,float angleY,float angleZ,
                                           float angXVariance ,float angYVariance , float angZVariance
                                         );


/**
* @brief Destroy and deallocate a photoshoot context created using createOGLRendererPhotoshootSandbox
* @ingroup OGLPhotoshoot
* @param Pointer to an OGL Photoshoot context created using createOGLRendererPhotoshootSandbox
* @retval 0=Failure,1=Success
*/
int destroyOGLRendererPhotoshootSandbox( void * photoConf );


/**
* @brief Get the center of a tile ( in order to extract its 2D rendering )
* @ingroup OGLPhotoshoot
* @param Pointer to an OGL Photoshoot context created using createOGLRendererPhotoshootSandbox
* @param Column of the rendering we want to get coordinates for
* @param Row of the rendering we want to get coordinates for
* @param Output X Coordinates
* @param Output Y Coordinates
* @retval 0=Failure,1=Success
*/
int getOGLPhotoshootTileXY(void * photoConf , unsigned int column , unsigned int row ,
                                              float * X , float * Y);
/**
* @brief Do a photoshoot on an object covering it from all viewpoints ( for object tracking )
* @bug The Photoshoot rendering mode is not ready yet , there need to be set rules to be able to reverse track the position of an object given
* @ingroup OGLPhotoshoot
* @param Pointer to an OGL Photoshoot context created using createOGLRendererPhotoshootSandbox
* @param The object id of the item to be photoshoot
* @param Number Of columns for the photoshoot
* @param Number Of rows for the photoshoot
* @param Distance at which the object should be remdered
* @param Central Rotation Angle on the X axis of the Object
* @param Central Rotation Angle on the Y axis of the Object
* @param Central Rotation Angle on the Z axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the X axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the Y axis of the Object
* @param Maximum Variance from the Central Rotation Angle on the Z axis of the Object
* @retval 0=Failure,1=Success
*/
int snapOGLRendererPhotoshootSandbox(
                                     void * photoConf ,
                                     int objID, unsigned int columns , unsigned int rows , float distance,
                                     float angleX,float angleY,float angleZ,
                                     float angXVariance ,float angYVariance , float angZVariance
                                    );


#ifdef __cplusplus
}
#endif

#endif // OGLRENDERERSANDBOX_H_INCLUDED
