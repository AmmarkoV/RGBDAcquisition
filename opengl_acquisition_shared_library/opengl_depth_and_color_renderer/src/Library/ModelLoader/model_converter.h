/** @file model_converter.h
 *  @brief  Converting through different internal representations..
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MODEL_CONVERTER_H_INCLUDED
#define MODEL_CONVERTER_H_INCLUDED

#include "model_loader_obj.h"
#include "model_loader_tri.h"

/**
* @brief Convert an OBJ file to a TRI file
* @ingroup conversions
* @param Output TRI File
* @param Input OBJ File
* @retval 0=Failure,1=Success
*/
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj);

#endif // MODEL_CONVERTER_H_INCLUDED
