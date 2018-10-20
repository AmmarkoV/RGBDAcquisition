#ifndef MODEL_EDITOR_H_INCLUDED
#define MODEL_EDITOR_H_INCLUDED


#include "model_loader_tri.h"
int punchHoleThroughModel(
                          struct TRI_Model * triModel ,
                          float * cylA ,
                          float * cylB ,
                          float radious,
                          float length
                          );

#endif // MODEL_EDITOR_H_INCLUDED
