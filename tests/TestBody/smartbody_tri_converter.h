/** @file smartbody_tri_converter.h
 *  @brief  A way to convert COCO models to SmartBody TRI
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef SMARTBODY_TRI_CONVERTER_H_INCLUDED
#define SMARTBODY_TRI_CONVERTER_H_INCLUDED


#include "../../tools/Primitives/skeleton.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_tri.h"


/**
* @brief Convert a coco skeleton to a smartbody TRI skeleton
* @ingroup SmartBody
* @param The input coco file
* @param The input triModel , after this paramter the bone information will be updated to match the coco model..!
* @param Output X position of the TRI model
* @param Output Y position of the TRI model
* @param Output Z position of the TRI model
* @param Output Quaternion qX param rotation of the TRI model
* @param Output Quaternion qY param rotation of the TRI model
* @param Output Quaternion qZ param rotation of the TRI model
* @param Output Quaternion qW param rotation of the TRI model
* @param The input triModel , after this paramter the bone information will be updated to match the coco model..!

* @retval 1=Error , 0=No Error
*/
int convertCOCO_To_Smartbody_TRI(const struct skeletonCOCO * coco,struct TRI_Model * triModel ,
                                 float *x , float *y , float *z ,
                                 float *qX,float *qY,float *qZ,float *qW );


#endif // SMARTBODY_TRI_CONVERTER_H_INCLUDED
