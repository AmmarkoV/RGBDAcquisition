#include "model_editor.h"

#include "../../../../tools/AmMatrix/matrixCalculations.h"


int punchHoleThroughModel(
                          struct TRI_Model * triModel ,
                          float * cylA ,
                          float * cylB ,
                          float radious,
                          float length
                          )
{
 unsigned int i=0;

 float res=0.0;
 float lengthsq = length * length ;
 float radius_sq = radious * radious ;




 for (i=0; i<triModel->header.numberOfVertices/3; i++)
 {
   res=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i*3+0] );

   if (res>0)
   {
      triModel->vertices[i*3+0]=0.0;
      triModel->vertices[i*3+1]=0.0;
      triModel->vertices[i*3+2]=0.0;
   }
 }


 return 0;
}
