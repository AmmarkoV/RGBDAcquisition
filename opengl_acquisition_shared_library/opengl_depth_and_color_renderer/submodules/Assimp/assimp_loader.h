#ifndef ASSIMP_LOADER_H_INCLUDED
#define ASSIMP_LOADER_H_INCLUDED


#include "../../src/ModelLoader/model_loader_tri.h"
#include "../../../../tools/Primitives/skeleton.h"

void deformOriginalModelAndBringBackFlatOneBasedOnThisSkeleton(
                                                                struct TRI_Model * outFlatModel ,
                                                                struct TRI_Model * inOriginalIndexedModel ,
                                                                struct skeletonHuman * sk
                                                              );

int testAssimp(const char * filename  , struct TRI_Model * triModel , struct TRI_Model * originalModel);


#endif // ASSIMP_LOADER_H_INCLUDED
