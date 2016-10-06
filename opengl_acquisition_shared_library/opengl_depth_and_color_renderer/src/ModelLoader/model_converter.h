#ifndef MODEL_CONVERTER_H_INCLUDED
#define MODEL_CONVERTER_H_INCLUDED

#include "model_loader_obj.h"
#include "model_loader_tri.h"

int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj);

#endif // MODEL_CONVERTER_H_INCLUDED
