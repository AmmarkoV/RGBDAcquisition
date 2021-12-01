#include "model_processor.h"
#include <stdio.h>
#include <stdlib.h>

#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"


void compressTRIModelToJointOnly(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN)
{
  fprintf(stderr,"compressTRIModelToJointOnly does not work correctly yet.. \n");

  tri_copyModel(triModelOUT,triModelIN,1,0);

  unsigned int outputNumberOfJoints=0;
  float * triJoints = convertTRIBonesToJointPositions(triModelOUT,&outputNumberOfJoints);
  if (triJoints!=0)
  {
    unsigned int  * verticesToKeep = getClosestVertexToJointPosition(triModelOUT,triJoints,outputNumberOfJoints);
    if (verticesToKeep!=0)
    {
      //----------------------

       //We want to only keep the vertices in the list so we will make a copy of the vertices and replace buffers one by one..!

       float * compressedNormals  = (float *) malloc (sizeof(float) * 3 *  outputNumberOfJoints * 3 /*We keep 3 copies of each vertice to make triangles..! */);
       float * compressedVertices = (float *) malloc (sizeof(float) * 3 *  outputNumberOfJoints * 3 /*We keep 3 copies of each vertice to make triangles..! */);
       unsigned int * compressedIndices = (unsigned int *) malloc (sizeof(unsigned int) * outputNumberOfJoints * 3 /*We keep 3 copies of each vertice to make triangles..! */);

        unsigned int i=0;
        for (i=0; i<outputNumberOfJoints; i++)
        {
          compressedVertices[i*3+0] = triModelOUT->vertices[verticesToKeep[i]+0];
          compressedVertices[i*3+1] = triModelOUT->vertices[verticesToKeep[i]+1];
          compressedVertices[i*3+2] = triModelOUT->vertices[verticesToKeep[i]+2];

          compressedNormals[i*3+0] = triModelOUT->normal[verticesToKeep[i]+0];
          compressedNormals[i*3+1] = triModelOUT->normal[verticesToKeep[i]+1];
          compressedNormals[i*3+2] = triModelOUT->normal[verticesToKeep[i]+2];

          compressedIndices[i*3+0] = i;
          compressedIndices[i*3+1] = i;
          compressedIndices[i*3+2] = i;
        }



      if ( triModelOUT->vertices!= 0 )      { free(triModelOUT->vertices); }
      if ( triModelOUT->normal!= 0 )        { free(triModelOUT->normal); }
      if ( triModelOUT->textureCoords!= 0 ) { free(triModelOUT->textureCoords); }
      if ( triModelOUT->colors!= 0 )        { free(triModelOUT->colors); }
      if ( triModelOUT->indices!= 0 )       { free(triModelOUT->indices); }

      triModelOUT->vertices = compressedVertices;
      triModelOUT->normal = compressedNormals;
      triModelOUT->indices  = compressedIndices;

      //----------------------
      free(verticesToKeep);
    }
    free(triJoints);
  }
 return;
}

