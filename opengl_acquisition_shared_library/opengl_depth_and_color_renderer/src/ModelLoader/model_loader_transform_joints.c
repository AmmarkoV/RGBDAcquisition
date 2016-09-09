#include "model_loader.h"
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"


int doModelTransform( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn )
{
  copyModelTri( triModelOut , triModelIn );



}


/*

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
