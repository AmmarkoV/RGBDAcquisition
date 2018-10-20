#include "model_editor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../../../tools/AmMatrix/matrixCalculations.h"

int punchHoleThroughModel(
                          struct TRI_Model * triModel ,
                          float * cylA ,
                          float * cylB ,
                          float radious,
                          float length
                          )
{
 char * selected = (char*) malloc(sizeof(char) * triModel->header.numberOfVertices);
 if (selected==0) { return 0; }

 memset(selected,0,sizeof(char) * triModel->header.numberOfVertices);

 unsigned int i=0 , indx=0;

 float res0=0.0,res1=0.0,res2=0.0;
 float lengthsq = length * length ;
 float radius_sq = radious * radious ;

 for (i=0; i<triModel->header.numberOfVertices/3; i++)
 {
   res1=pointIsInsideCylinder( cylA , cylB , lengthsq , radius_sq , &triModel->vertices[i*3+0] );

   if (res1>0)
   {
      selected[i]=1;
   }
 }

 unsigned int i0,i1,i2;

 for (indx=0; indx<triModel->header.numberOfIndices/3; indx++)
 {
   i0=triModel->indices[indx*3+0];
   i1=triModel->indices[indx*3+1];
   i2=triModel->indices[indx*3+2];

   if ( selected[i0] )   { res0=1; } else {res0=0;}
   if ( selected[i1] )   { res1=1; } else {res1=0;}
   if ( selected[i2] )   { res2=1; } else {res2=0;}


   if ( (res0) || (res1) || (res2) )
    {
      unsigned int oV = i1;

      if (!res0) { oV=i0; } else
      if (!res1) { oV=i1; } else
      if (!res2) { oV=i2; }

      triModel->indices[indx*3+0]=oV;
      triModel->indices[indx*3+1]=oV;
      triModel->indices[indx*3+2]=oV;
    }
 }


 free(selected);

 return 0;
}

