#include "model_loader.h"
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"




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



void transformMeshBasedOnSkeleton(struct aiScene *scene , int meshNumber , struct TRI_Model * indexed , struct skeletonHuman * sk )
{

    //The goal here is to transform the mesh stored int indexed using a skeleton stored in sk
    struct aiMesh * mesh = scene->mMeshes[meshNumber];
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);


    //The first step to do is create an internal structure and gather all the information out of Assimp
    struct boneState modifiedSkeleton;
    populateInternalRigState(scene , meshNumber, &modifiedSkeleton );


    //After we have it we can now use it to read the node heirarchy
    aiMatrix4x4 Identity;
    aiMakeIdentity(&Identity);
    readNodeHeirarchyOLD(mesh,scene->mRootNode,&modifiedSkeleton,sk,Identity,0);

    //We NEED to clear the vertices and normals since they are added uppon , not having
    //the next two lines results in really weird and undebuggable visual behaviour
	memset(indexed->vertices, 0, indexed->header.numberOfVertices  * sizeof(float));
	memset(indexed->normal  , 0, indexed->header.numberOfNormals   * sizeof(float));

    unsigned int i=0,k=0;
	for (k = 0; k < modifiedSkeleton.numberOfBones; k++)
    {
	   struct aiBone *bone = mesh->mBones[k];
	   struct aiNode *node = findNode(scene->mRootNode, bone->mName.data);

       //Update all vertices with current weighted transforms for current the current bone
       //stored in skin3 and skin4
       aiMatrix3x3 finalTransform3x3;
	   extract3x3(&finalTransform3x3, &modifiedSkeleton.bone[k].finalTransform);
       for (i = 0; i < bone->mNumWeights; i++)
        {
			unsigned int v = bone->mWeights[i].mVertexId;
			float w = bone->mWeights[i].mWeight;

			aiVector3D position = mesh->mVertices[v];
			aiTransformVecByMatrix4(&position, &modifiedSkeleton.bone[k].finalTransform );
			indexed->vertices[v*3+0] += (float) position.x * w;
			indexed->vertices[v*3+1] += (float) position.y * w;
			indexed->vertices[v*3+2] += (float) position.z * w;

			aiVector3D normal = mesh->mNormals[v];
			aiTransformVecByMatrix3(&normal, &finalTransform3x3);
			indexed->normal[v*3+0] += (float) normal.x * w;
			indexed->normal[v*3+1] += (float) normal.y * w;
			indexed->normal[v*3+2] += (float) normal.z * w;
		}
	}
}
*/






int doModelTransform( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn , float * jointData , unsigned int jointDataSize)
{
  copyModelTri( triModelOut , triModelIn );


   unsigned int i=0;
   for (i=0; i<triModelIn->header.numberOfBones; i++ )
   {

   }


}

