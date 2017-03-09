#include "model_processor.h"


void compressTRIModelToJointOnly(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN)
{
  copyModelTri(triModelOUT,triModelIN,1);

  unsigned int outputNumberOfJoints;
  float * triJoints = convertTRIBonesToJointPositions(triModelOUT,&outputNumberOfJoints);
  if (triJoints!=0)
  {
    unsigned int  * verticesToKeep = getClosestVertexToJointPosition(triModelOUT,triJoints,outputNumberOfJoints);
    if (verticesToKeep!=0)
    {
      //----------------------

       //We want to only keep the vertices in the list so we will make a copy of the vertices and replace buffers one by one..!

       float * compressedVertices = (float *) malloc (sizeof(float) * 3 *  outputNumberOfJoints * 3 /*We keep 3 copies of each vertice to make triangles..! */);
       unsigned int * compressedIndices = (unsigned int *) malloc (sizeof(unsigned int) * outputNumberOfJoints * 3 /*We keep 3 copies of each vertice to make triangles..! */);

        unsigned int i=0;
        for (i=0; i<outputNumberOfJoints; i++)
        {
          compressedVertices[i*3+0] = triModelOUT->vertices[verticesToKeep[i]+0];
          compressedVertices[i*3+1] = triModelOUT->vertices[verticesToKeep[i]+1];
          compressedVertices[i*3+2] = triModelOUT->vertices[verticesToKeep[i]+2];

          compressedIndices[i*3+0] = i;
          compressedIndices[i*3+1] = i;
          compressedIndices[i*3+2] = i;
        }




      //----------------------
      free(verticesToKeep);
    }
    free(triJoints);
  }
 return;
}

