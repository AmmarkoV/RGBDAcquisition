#ifndef TILEDRENDERER_H_INCLUDED
#define TILEDRENDERER_H_INCLUDED


#include "TrajectoryParser/TrajectoryParser.h"
#include "model_loader.h"
#include "scene.h"


int getPhotoshootTile2DCoords(unsigned int column, unsigned int row , double * x2D , double *y2D , double * z2D);


int tiledRenderer_Render(
                             struct VirtualStream * scene  ,
                             struct Model ** models ,
                             int objID,unsigned int columns , unsigned int rows , float distance,
                             float angleX,float angleY,float angleZ ,
                             float angXVariance ,float angYVariance , float angZVariance
                         );

#endif // TILEDRENDERER_H_INCLUDED
