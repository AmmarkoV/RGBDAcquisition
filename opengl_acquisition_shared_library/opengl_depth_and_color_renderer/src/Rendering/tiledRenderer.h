/** @file tiledRenderer.h
 *  @brief  Rendering multiple
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef TILEDRENDERER_H_INCLUDED
#define TILEDRENDERER_H_INCLUDED


#include "../TrajectoryParser/TrajectoryParser.h"
#include "../ModelLoader/model_loader.h"
#include "../Scene/scene.h"


/**
* @brief An enumerator to represent position coordinates
*/
enum POS_COORDS
{
    POS_X=0,
    POS_Y,
    POS_Z,
    POS_ANGLEX,
    POS_ANGLEY,
    POS_ANGLEZ,
    POS_COORD_LENGTH
};


/**
* @brief Depending on these values we get more or less tiles with more or less pixels per tile
*        So this has a direct impact on the information provided
*/
struct tiledRendererDetail
{
 float OGLUnitWidth , OGLUnitHeight ;

 unsigned int snapsHorizontal , snapsVertical;

 float posOffsetX , posOffsetY ;

 float posXBegining , posYBegining;

 float angXStep , angYStep , angZStep ;

};


/**
* @brief Tiled rendering configuration
*/
struct tiledRendererConfiguration
{
  void * scenePTR;
  void * modelStoragePTR;
  unsigned int objID;

  unsigned int columns;
  unsigned int rows;
  float distance , angleX, angleY, angleZ , angXVariance , angYVariance , angZVariance;


  struct tiledRendererDetail op;



};



/**
* @brief Get The 2D center for the rendering on specific Column and Row
* @ingroup TiledRenderer
* @param Tiled Renderer Context
* @param Column to get info about
* @param Row to get info about
* @param X in the 2D Rendered output
* @param Y in the 2D Rendered output
* @param Z ( depth value ) of the center pixel
* @retval 1=Success , 0=Failure
*/
int tiledRenderer_get2DCenter(void * trConf ,
                              unsigned int column, unsigned int row ,
                              float * x2D , float *y2D , float * z2D);






/**
* @brief Render one shot
* @ingroup TiledRenderer
* @param Tiled Renderer Context
* @retval 1=Success , 0=Failure
*/
int tiledRenderer_Render( struct tiledRendererConfiguration * trConf);




#endif // TILEDRENDERER_H_INCLUDED
