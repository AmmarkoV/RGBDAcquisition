#ifndef ASSIMP_LOADER_H_INCLUDED
#define ASSIMP_LOADER_H_INCLUDED


#include "../../src/ModelLoader/model_loader_tri.h"

int convertAssimpToTRI(const char * filename  , struct TRI_Model * triModel , struct TRI_Model * originalModel);


#endif // ASSIMP_LOADER_H_INCLUDED
