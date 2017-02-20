#ifndef SMARTBODY_TRI_CONVERTER_H_INCLUDED
#define SMARTBODY_TRI_CONVERTER_H_INCLUDED


#include "../../tools/Primitives/skeleton.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_tri.h"

int convertCOCO_To_Smartbody_TRI(struct skeletonCOCO * coco,struct TRI_Model * triModel ,
                                 float *x , float *y , float *z ,
                                 float *qX,float *qY,float *qZ,float *qW );


#endif // SMARTBODY_TRI_CONVERTER_H_INCLUDED
