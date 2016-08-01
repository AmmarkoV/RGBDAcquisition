#include <stdio.h>
#include "../../src/ModelLoader/model_loader_tri.h"
#include "assimp_loader.h"

int main (int argc, char *argv[])
{
 struct TRI_Model flatModel;
 struct TRI_Model originalModel;

 testAssimp(argv[1],&flatModel,&originalModel);


 saveModelTri(argv[2], &originalModel);

 freeModelTri(&flatModel);
 freeModelTri(&originalModel);

 return 0;

}
