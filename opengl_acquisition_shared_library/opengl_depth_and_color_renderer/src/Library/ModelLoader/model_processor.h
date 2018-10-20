/** @file model_processor.h
 *  @brief  TRIModels basic processing functions functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            The basic idea is a 3D mesh container that
            1) Contains Vertices/Colors/TextureCoords/Normals  indexed or not
            2) Contains skinning information
            3) Is a binary format , so ( smaller filesizes )
            4) Does not have dependencies ( plain C code )
            5) Is as small and clear as possible
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MODEL_PROCESSOR_H_INCLUDED
#define MODEL_PROCESSOR_H_INCLUDED

#include "model_loader_tri.h"



void compressTRIModelToJointOnly(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN);


#endif // MODEL_PROCESSOR_H_INCLUDED
