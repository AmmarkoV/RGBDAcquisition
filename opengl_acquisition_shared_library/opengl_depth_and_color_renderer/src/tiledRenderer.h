#ifndef TILEDRENDERER_H_INCLUDED
#define TILEDRENDERER_H_INCLUDED


#include "TrajectoryParser/TrajectoryParser.h"
#include "model_loader.h"
#include "scene.h"


struct tiledRendererConfiguration
{
  void * scenePTR;
  void * modelPTR;
  unsigned int objID;

  unsigned int columns;
  unsigned int rows;
  float distance , angleX, angleY, angleZ , angXVariance , angYVariance , angZVariance;
};


int getPhotoshootTile2DCoords(unsigned int column, unsigned int row , double * x2D , double *y2D , double * z2D);


int tiledRenderer_Render( struct tiledRendererConfiguration * configuration);

#endif // TILEDRENDERER_H_INCLUDED
