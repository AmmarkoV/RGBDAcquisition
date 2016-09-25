#include "model_loader.h"
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"

#include <stdio.h>
#include <stdlib.h>



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
/*



void readNodeHeirarchyOLD(const aiMesh * mesh , const aiNode* pNode,  struct boneState * bones , struct skeletonHuman * sk, aiMatrix4x4 & ParentTransform , unsigned int recursionLevel)
{

    if (recursionLevel==0)    { fprintf(stderr,"readNodeHeirarchy : \n"); } else
                              {  fprintf(stderr,"   "); }
    fprintf(stderr,"%s\n" , pNode->mName.data);

    aiMatrix4x4 NodeTransformation=pNode->mTransformation;


    unsigned int foundBone;
    unsigned int boneNumber=findBoneFromString(mesh,pNode->mName.data,&foundBone);


    unsigned int i=0;
    if (foundBone)
    {
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            if (strcmp(pNode->mName.data , smartBodyNames[i])==0)
              {
               if ( sk->active[i] )
               {
               fprintf(stderr,GREEN "hooked with %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" NORMAL,jointNames[i] , sk->relativeJointAngle[i].x, sk->relativeJointAngle[i].y, sk->relativeJointAngle[i].z);
               bones->bone[boneNumber].ScalingVec.x=1.0;
               bones->bone[boneNumber].ScalingVec.y=1.0;
               bones->bone[boneNumber].ScalingVec.z=1.0;

               bones->bone[boneNumber].TranslationVec.x=pNode->mTransformation.a4;
               bones->bone[boneNumber].TranslationVec.y=pNode->mTransformation.b4;
               bones->bone[boneNumber].TranslationVec.z=pNode->mTransformation.c4;

              aiMatrix4x4::Scaling(bones->bone[boneNumber].ScalingVec,bones->bone[boneNumber].scalingMat);
              aiMatrix4x4::Translation (bones->bone[boneNumber].TranslationVec,bones->bone[boneNumber].translationMat);
              //aiMakeQuaternion( &bones.bone[k].rotationMat , &bones.bone[k].RotationQua );
              //aiPrintMatrix(&bones->bone[boneNumber].rotationMat );


               //zxy 120 - xyz 012

               bones->bone[boneNumber].rotationMat.FromEulerAnglesXYZ(
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].z + defaultJointsOffsetZXY[i*3+2] ),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].x + defaultJointsOffsetZXY[i*3+0] ),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].y + defaultJointsOffsetZXY[i*3+1] )
                                                                      );


               NodeTransformation =  bones->bone[boneNumber].translationMat  * bones->bone[boneNumber].rotationMat * bones->bone[boneNumber].scalingMat;
              } else
              {
               fprintf(stderr, RED " inactive %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" NORMAL ,jointNames[i] ,
                       sk->relativeJointAngle[i].x,
                       sk->relativeJointAngle[i].y,
                       sk->relativeJointAngle[i].z);
              }
            }
        }

    aiMatrix4x4 GlobalTransformation = ParentTransform  * NodeTransformation;
    bones->bone[boneNumber].finalTransform = m_GlobalInverseTransform * GlobalTransformation * bones->bone[boneNumber].boneInverseBindTransform;
    for ( i = 0 ; i < pNode->mNumChildren ; i++)
    {
        readNodeHeirarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
    }
    } else
    {
      aiMatrix4x4 GlobalTransformation = ParentTransform  * pNode->mTransformation;
      fprintf(stderr,"        <!%s!>\n",pNode->mName.data);
       for ( i = 0 ; i < pNode->mNumChildren ; i++)
       {
         readNodeHeirarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
       }
    }
}


*/








void recursiveJointHeirarchyTransformer(
                                         struct TRI_Model * in  ,
                                         int curBone ,
                                         struct TRI_Transform * finalTransforms ,
                                         double * parentTransform ,
                                         float * jointData , unsigned int jointDataSize ,
                                         unsigned int recursionLevel
                                       )
{
    unsigned int doDBGPrintout=0;
    if (recursionLevel==40) { fprintf(stderr,RED "_____________________\n BUG : REACHED RECURSION LIMIT \n_____________________\n" NORMAL); return; }
    unsigned int i=0;


    if (in->bones[curBone].info->altered)  { doDBGPrintout=1; }


    if (doDBGPrintout)
    {
    if (recursionLevel==0)    { fprintf(stderr,"readNodeHeirarchy : \n"); } else
                              { fprintf(stderr,"   "); }
     fprintf(stderr,"(%u) " , curBone );
     fprintf(stderr,"%s\n" , in->bones[curBone].boneName );

     if (in->bones[curBone].info->altered) { fprintf(stderr,GREEN "hooked  \n" NORMAL );  } else
                                           { fprintf(stderr, RED " inactive  \n" NORMAL); }
    }


    double nodeTransformation[16];
    create4x4IdentityMatrix(&nodeTransformation) ;


    double * globalTransformation[16];
    create4x4IdentityMatrix(&globalTransformation);


  if (in->bones[curBone].info->altered)
    {
      //aiMatrix4x4 GlobalTransformation = ParentTransform  * NodeTransformation;
      multiplyTwo4x4Matrices(&globalTransformation,parentTransform,&nodeTransformation);


      double * finalMatrix = finalTransforms[i].finalTransform;
      multiplyThree4x4Matrices( finalMatrix, &globalTransformation,parentTransform,&nodeTransformation);
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
      //aiMatrix4x4 GlobalTransformation = ParentTransform  * pNode->mTransformation;
      multiplyTwo4x4Matrices(&globalTransformation,parentTransform,&nodeTransformation);


      //fprintf(stderr,"%s has %u children \n" , in->bones[curBone].boneName , in->bones[curBone].info->numberOfBoneChildren );
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
  if (triModelIn==0) { fprintf(stderr,"doModelTransform called without input TRI Model \n"); return 0; }

  //Reduce spam
  //printTRIBoneStructure(triModelIn,0 /*Dont show all matrices*/);
  if ( ( triModelIn->vertices ==0 ) || ( triModelIn->header.numberOfVertices ==0 ) )
  {
     fprintf(stderr,RED "Number of vertices is zero so can't do model transform using weights..\n" NORMAL);
    return 0;
  }

  copyModelTri( triModelOut , triModelIn , 1 /*We also want bone data*/);
  //fprintf(stderr,RED "will not perform transform because of DEBUG..! \n" NORMAL);
  //return 1; //to test if copy works ok

  double transPosition[4]={0}; double transNormal[4]={0};
  double position[4]={0}; double normal[4]={0};


   struct TRI_Transform * finalTransforms = (struct TRI_Transform * ) malloc( triModelIn->header.numberOfBones * sizeof(struct TRI_Transform) );
   if (finalTransforms==0) { return 0; }
   unsigned int i=0;

   double * finalMatrix;
   for (i=0; i<triModelIn->header.numberOfBones; i++)
   {
     //fprintf(stderr,"Final Matrix %u \n" , i);
     finalMatrix = finalTransforms[i].finalTransform;
     create4x4IdentityMatrix(finalMatrix);
     //print4x4DMatrix(" data ", finalMatrix);
   }


  double parentTransform[16]={0};
  create4x4IdentityMatrix(&parentTransform) ;

  unsigned int rootBone = 0;
  findTRIBoneWithName(triModelIn,"JtRoot",&rootBone);
  recursiveJointHeirarchyTransformer( triModelIn , rootBone , finalTransforms , parentTransform , jointData , jointDataSize , 0 );


   //fprintf(stderr,"Clearing vertices & normals \n");
   //We NEED to clear the vertices and normals since they are added uppon , not having
   //the next two lines results in really weird and undebuggable visual behaviour
   memset(triModelOut->vertices, 0, triModelOut->header.numberOfVertices  * sizeof(float));
   memset(triModelOut->normal  , 0, triModelOut->header.numberOfNormals   * sizeof(float));



   fprintf(stderr,"Transforming bones : ");
   unsigned int k=0;
   for (k=0; k<triModelIn->header.numberOfBones; k++ )
   {
     //fprintf(stderr,"%u ",k);
     //if (is4x4DIdentityMatrix(finalTransforms[k].finalTransform ))
     //{ fprintf(stderr,"Has identity transform \n"); }


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

