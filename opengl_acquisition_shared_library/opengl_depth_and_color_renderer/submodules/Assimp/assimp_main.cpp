#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/ModelLoader/model_loader_tri.h"
#include "assimp_loader.h"

int main (int argc, char *argv[])
{
 if (argc<4)   { fprintf(stderr,"Not enough arguments , use as : \nassimpTester --convert source.dae target.tri \n"); return 0; }


 fprintf(stderr,"assimpTester %s %s %s \n",argv[1],argv[2],argv[3]);


 struct TRI_Model *flatModel    =  allocateModelTri();
 struct TRI_Model *originalModel=  allocateModelTri();


 convertAssimpToTRI(argv[2],flatModel,originalModel);
 saveModelTri(argv[3], originalModel);



   if (strcmp(argv[1],"--test")==0)
   {
    struct TRI_Model *reloadedModel=  allocateModelTri();
    loadModelTri( argv[3] , reloadedModel);
    saveModelTri( "resave.tri", reloadedModel);

    freeModelTri(reloadedModel);
   }

 freeModelTri(flatModel);
 freeModelTri(originalModel);

 return 0;

}
