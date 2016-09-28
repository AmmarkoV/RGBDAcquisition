#include "model_loader.h"
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "../../../../tools/AmMatrix/matrixCalculations.h"
#include "../../../../tools/AmMatrix/matrix4x4Tools.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

void recursiveJointHeirarchyTransformer(
                                         struct TRI_Model * in  ,
                                         int curBone ,
                                         struct TRI_Transform * finalTransforms ,
                                         double * parentTransform ,
                                         float * jointData , unsigned int jointDataSize ,
                                         unsigned int recursionLevel
                                       )
{
    if (recursionLevel>=40) { fprintf(stderr,RED "_____________________\n BUG : REACHED RECURSION LIMIT \n_____________________\n" NORMAL); return; }
    unsigned int i=0;

    double globalTransformation[16] , nodeTransformation[16];
    create4x4IdentityMatrix(nodeTransformation) ;
    create4x4IdentityMatrix(globalTransformation);


  if (in->bones[curBone].info->altered)
    {
      //aiMatrix4x4 GlobalTransformation = ParentTransform  * NodeTransformation;
      print4x4DMatrixTRI("mTransformation was .. \n",in->bones[curBone].info->boneTransformation);

      double translation[16] , rotation[16] , scaling[16];
      create4x4IdentityMatrix(translation) ;
      create4x4IdentityMatrix(rotation);
      create4x4IdentityMatrix(scaling);

      //Get Translation
      translation[3] =in->bones[curBone].info->boneTransformation[3];
      translation[7] =in->bones[curBone].info->boneTransformation[7];
      translation[11]=in->bones[curBone].info->boneTransformation[11];

      multiplyThree4x4Matrices( nodeTransformation, translation,rotation,scaling);


      print4x4DMatrixTRI("Translation was .. ",translation);
      print4x4DMatrixTRI("Scaling was .. ",scaling);
      print4x4DMatrixTRI("Rotation was .. ",rotation);
      print4x4DMatrixTRI("Node Transformation is now.. \n",nodeTransformation);

      multiplyTwo4x4Matrices(globalTransformation,parentTransform,nodeTransformation);

      multiplyThree4x4Matrices( finalTransforms[i].finalTransform, globalTransformation,parentTransform,nodeTransformation);
      //bones->bone[boneNumber].finalTransform = m_GlobalInverseTransform * GlobalTransformation * bones->bone[boneNumber].boneInverseBindTransform;
     for ( i = 0 ; i < in->bones[curBone].info->numberOfBoneChildren; i++)
      {
        //readNodeHeirarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
        unsigned int curBoneChild=in->bones[curBone].info->boneChild[i];
        recursiveJointHeirarchyTransformer(
                                           in  ,
                                           curBoneChild ,
                                           finalTransforms ,
                                           parentTransform ,
                                           jointData , jointDataSize ,
                                           recursionLevel+1
                                         );
      }
    } else
    {
      fprintf(stderr,"Unedited bone %u " , curBone);
      fprintf(stderr,"%s has %u children \n" , in->bones[curBone].boneName , in->bones[curBone].info->numberOfBoneChildren );
            for ( i = 0 ; i < in->bones[curBone].info->numberOfBoneChildren; i++)
       { fprintf(stderr," %s " , in->bones[in->bones[curBone].info->boneChild[i]].boneName );   }
      fprintf(stderr,"\n");

      //aiMatrix4x4 GlobalTransformation = ParentTransform  * pNode->mTransformation;
      multiplyTwo4x4Matrices(globalTransformation,parentTransform,nodeTransformation);


      for ( i = 0 ; i < in->bones[curBone].info->numberOfBoneChildren; i++)
       {
        // readNodeHeirarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
        unsigned int curBoneChild=in->bones[curBone].info->boneChild[i];

        //fprintf(stderr," recursing children %s (%u/%u) " , in->bones[curBoneChild].boneName , i, in->bones[curBone].info->numberOfBoneChildren );
        recursiveJointHeirarchyTransformer(
                                           in  ,
                                           curBoneChild ,
                                           finalTransforms ,
                                           parentTransform ,
                                           jointData , jointDataSize ,
                                           recursionLevel+1
                                         );
       }
    }

}




int doModelTransform( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn , float * jointData , unsigned int jointDataSize)
{
  if (triModelIn==0)
                     { fprintf(stderr,"doModelTransform called without input TRI Model \n"); return 0; }
  if ( ( triModelIn->vertices ==0 ) || ( triModelIn->header.numberOfVertices ==0 ) )
                     { fprintf(stderr,RED "Number of vertices is zero so can't do model transform using weights..\n" NORMAL); return 0; }
  //Past checks..

 struct TRI_Transform * finalTransforms = (struct TRI_Transform * ) malloc( triModelIn->header.numberOfBones * sizeof(struct TRI_Transform) );
 if (finalTransforms==0) { return 0; }

 copyModelTri( triModelOut , triModelIn , 1 /*We also want bone data*/);

 double transPosition[4]={0} ,transNormal[4]={0} , position[4]={0} , normal[4]={0};


 unsigned int i=0;
 for (i=0; i<triModelIn->header.numberOfBones; i++)
   {
     double * finalMatrix = finalTransforms[i].finalTransform;
     //create4x4IdentityMatrix(finalMatrix);
     unsigned int k=0;
     for (k=0; k<16; k++)
     {
       finalMatrix[k] = jointData[i*16+k];
     }

     if (!is4x4DIdentityMatrix(finalMatrix))
        { triModelIn->bones[i].info->altered=1; } else
        { triModelIn->bones[i].info->altered=0; }
   }


  double parentTransform[16]={0};
  create4x4IdentityMatrix(parentTransform) ;

   //This recursively calculates all matrix transforms and prepares the correct matrices
   //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     recursiveJointHeirarchyTransformer( triModelIn , triModelIn->header.rootBone , finalTransforms , parentTransform , jointData , jointDataSize , 0 );
   //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  //We NEED to clear the vertices and normals since they are added uppon , not having
  //the next two lines results in really weird and undebuggable visual behaviour
  memset(triModelOut->vertices, 0, triModelOut->header.numberOfVertices  * sizeof(float));
  memset(triModelOut->normal  , 0, triModelOut->header.numberOfNormals   * sizeof(float));



   fprintf(stderr,"Transforming bones : ");
   unsigned int k=0;
   for (k=0; k<triModelIn->header.numberOfBones; k++ )
   {
     if (triModelIn->bones[k].info->altered)
          { fprintf(stderr,"%u (%s) altered \n",k , triModelIn->bones[k].boneName); }


     for (i=0; i<triModelIn->bones[k].info->boneWeightsNumber; i++ )
     {
       //V is the vertice we will be working in this loop
       unsigned int v = triModelIn->bones[k].weightIndex[i];
       //W is the weight that we have for the specific bone
       float w = triModelIn->bones[k].weightValue[i];

       //We load our input into position/normal
       position[0] = triModelIn->vertices[v*3+0];
       position[1] = triModelIn->vertices[v*3+1];
       position[2] = triModelIn->vertices[v*3+2];
       position[3] = 1.0;

       normal[0]   = triModelIn->normal[v*3+0];
       normal[1]   = triModelIn->normal[v*3+1];
       normal[2]   = triModelIn->normal[v*3+2];
       normal[3]   = 1.0;

       //We transform input (initial) position with the transform we computed to get transPosition
       transform3DPointVectorUsing4x4Matrix(transPosition, finalTransforms[k].finalTransform ,position);
	   triModelOut->vertices[v*3+0] += (float) transPosition[0] * w;
	   triModelOut->vertices[v*3+1] += (float) transPosition[1] * w;
	   triModelOut->vertices[v*3+2] += (float) transPosition[2] * w;

       //We transform input (initial) normal with the transform we computed to get transNormal
       transform3DPointVectorUsing4x4Matrix(transNormal, finalTransforms[k].finalTransform ,normal);
	   triModelOut->normal[v*3+0] += (float) transNormal[0] * w;
	   triModelOut->normal[v*3+1] += (float) transNormal[1] * w;
	   triModelOut->normal[v*3+2] += (float) transNormal[2] * w;
     }
   }
   fprintf(stderr," done \n");


  free(finalTransforms);
 return 1;
}

