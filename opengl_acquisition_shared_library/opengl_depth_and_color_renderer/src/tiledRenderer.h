#ifndef TILEDRENDERER_H_INCLUDED
#define TILEDRENDERER_H_INCLUDED


#include "TrajectoryParser/TrajectoryParser.h"
#include "model_loader.h"
#include "scene.h"


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


struct tiledRendererDetail
{
 float OGLUnitWidth , OGLUnitHeight ;

 unsigned int snapsHorizontal , snapsVertical;

 float posOffsetX , posOffsetY ;

 float posXBegining , posYBegining;

 float angXStep , angYStep , angZStep ;



};


struct tiledRendererConfiguration
{
  void * scenePTR;
  void * modelPTR;
  unsigned int objID;

  unsigned int columns;
  unsigned int rows;
  float distance , angleX, angleY, angleZ , angXVariance , angYVariance , angZVariance;


  struct tiledRendererDetail op;



};



int tiledRenderer_get2DCenter(void * trConf ,
                              unsigned int column, unsigned int row ,
                              float * x2D , float *y2D , float * z2D);

int tiledRenderer_Render( struct tiledRendererConfiguration * trConf);




#endif // TILEDRENDERER_H_INCLUDED
