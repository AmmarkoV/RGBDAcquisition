#include "model_editor.h"

#include "../../../../tools/AmMatrix/matrixCalculations.h"


#define OUT 0.0

int punchHoleThroughModel(
                          struct TRI_Model * triModel ,
                          float * cylA ,
                          float * cylB ,
                          float radious,
                          float length
                          )
{
 unsigned int i=0 , indx=0;

 float res0=0.0,res1=0.0,res2=0.0;
 float lengthsq = length * length ;
 float radius_sq = radious * radious ;

 for (i=0; i<triModel->header.numberOfVertices/3; i++)
 {
   res1=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i*3+0] );

   if (res1>0)
   {
      triModel->vertices[i*3+0]=OUT;
      triModel->vertices[i*3+1]=OUT;
      triModel->vertices[i*3+2]=OUT;
   }
 }


return 0;

 unsigned int i0,i1,i2;

 for (indx=0; indx<triModel->header.numberOfIndices/3; indx++)
 {
   i0=triModel->indices[indx*3+0];
   i1=triModel->indices[indx*3+1];
   i2=triModel->indices[indx*3+2];

  // res1=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i1] );
  // res2=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i2] );
  // res3=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i3] );

   if ( (triModel->vertices[i0+0]==OUT) && (triModel->vertices[i0+1]==OUT) &&  (triModel->vertices[i0+2]==OUT) )   { res0=1; } else {res0=0;}
   if ( (triModel->vertices[i1+0]==OUT) && (triModel->vertices[i1+1]==OUT) &&  (triModel->vertices[i1+2]==OUT) )   { res1=1; } else {res1=0;}
   if ( (triModel->vertices[i2+0]==OUT) && (triModel->vertices[i2+1]==OUT) &&  (triModel->vertices[i2+2]==OUT) )   { res2=1; } else {res2=0;}


   if ( (res0) && (!res1) && (!res2) ) { triModel->indices[indx*3+0]=i1; }
   if ( (!res0) && (res1) && (!res2) ) { triModel->indices[indx*3+1]=i2; }
   if ( (!res0) && (!res1) && (res2) ) { triModel->indices[indx*3+2]=i0; }


   if ( (res0) && (res1) && (!res2) ) { triModel->indices[indx*3+0]=i2;
                                        triModel->indices[indx*3+1]=i2;   }
   if ( (!res0) && (res1) && (res2) ) { triModel->indices[indx*3+1]=i0;
                                        triModel->indices[indx*3+2]=i0;   }
   if ( (res0) && (res1) && (!res2) ) { triModel->indices[indx*3+0]=i2;
                                        triModel->indices[indx*3+1]=i2;   }


 }


 return 0;
}


















