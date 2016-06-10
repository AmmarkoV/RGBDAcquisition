#include "assimp_loader.h"


#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#include "skeleton.h"

#define DO_TRANSFORM 0
#define USE_NEW_TRANSFORM_MESH_CODE 1



struct boneItem
{
  aiMatrix4x4 finalTransform;
  aiMatrix4x4 parentTransform;
  aiMatrix4x4 nodeTransform;
  aiMatrix4x4 globalTransform;
  aiMatrix4x4 boneBindTransform;

  int tampered;

  aiMatrix4x4 scalingMat;
  aiMatrix4x4 translationMat;
  aiMatrix4x4 rotationMat;
  aiMatrix4x4 finalMat;

  aiVector3D ScalingVec;
  aiVector3D TranslationVec;
  aiQuaternion RotationQua;

  char name[128];
  int itemID;
  int parentItemID;
  int parentlessNode;
 };

struct boneState
{
  struct boneItem bone[100];
  unsigned int numberOfBones;

};

struct aiScene *g_scene = NULL;
aiMatrix4x4 m_GlobalInverseTransform;
struct boneState modifiedSkeleton;



void extract3x3( aiMatrix3x3 *m3,  aiMatrix4x4 *m4)
{
	m3->a1 = m4->a1; m3->a2 = m4->a2; m3->a3 = m4->a3;
	m3->b1 = m4->b1; m3->b2 = m4->b2; m3->b3 = m4->b3;
	m3->c1 = m4->c1; m3->c2 = m4->c2; m3->c3 = m4->c3;
}




void aiMakeQuaternion(aiMatrix4x4 * am , aiQuaternion * qu)
{
    float yy2 = 2.0f * qu->y * qu->y;
    float xy2 = 2.0f * qu->x * qu->y;
    float xz2 = 2.0f * qu->x * qu->z;
    float yz2 = 2.0f * qu->y * qu->z;
    float zz2 = 2.0f * qu->z * qu->z;
    float wz2 = 2.0f * qu->w * qu->z;
    float wy2 = 2.0f * qu->w * qu->y;
    float wx2 = 2.0f * qu->w * qu->x;
    float xx2 = 2.0f * qu->x * qu->x;
    am->a1 = - yy2 - zz2 + 1.0f;
    am->a2 = xy2 + wz2;
    am->a3 = xz2 - wy2;
    am->a4 = 0;
    am->b1 = xy2 - wz2;
    am->b2 = - xx2 - zz2 + 1.0f;
    am->b3 = yz2 + wx2;
    am->b4 = 0;
    am->c1 = xz2 + wy2;
    am->c2 = yz2 - wx2;
    am->c3 = - xx2 - yy2 + 1.0f;
    am->c4 = 0.0f;
    am->d1 = 0.0;
    am->d2 = 0.0;
    am->d3 = 0.0;
    am->d4 = 1.0f;
}

void aiMakeIdentity(aiMatrix4x4 * am)
{
 am->a1 = 1.0; am->a2 = 0.0;  am->a3 = 0.0;  am->a4 = 0.0;
 am->b1 = 0.0; am->b2 = 1.0;  am->b3 = 0.0;  am->b4 = 0.0;
 am->c1 = 0.0; am->c2 = 0.0;  am->c3 = 1.0;  am->c4 = 0.0;
 am->d1 = 0.0; am->d2 = 0.0;  am->d3 = 0.0;  am->d4 = 1.0;
}


int aiMatricesSame(aiMatrix4x4 * am , aiMatrix4x4 * bm)
{
 unsigned int similarity = 0;
 similarity += (am->a1==bm->a1);
 similarity += (am->a2==bm->a2);
 similarity += (am->a3==bm->a3);
 similarity += (am->a4==bm->a4);

 similarity += (am->b1==bm->b1);
 similarity += (am->b2==bm->b2);
 similarity += (am->b3==bm->b3);
 similarity += (am->b4==bm->b4);

 similarity += (am->c1==bm->c1);
 similarity += (am->c2==bm->c2);
 similarity += (am->c3==bm->c3);
 similarity += (am->c4==bm->c4);

 similarity += (am->d1==bm->d1);
 similarity += (am->d2==bm->d2);
 similarity += (am->d3==bm->d3);
 similarity += (am->d4==bm->d4);

 return (similarity==16);
}



void aiPrintMatrix(aiMatrix4x4 * am)
{
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->a1, am->a2,  am->a3,   am->a4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->b1, am->b2,  am->b3,   am->b4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->c1, am->c2,  am->c3,   am->c4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->d1, am->d2,  am->d3,   am->d4 );
}


float rad_to_degrees(float radians)
{
    return radians * (180.0 / M_PI);
}

float degrees_to_rad(float degrees)
{
    return degrees * (M_PI /180.0 );
}



// find a node by name in the hierarchy (for anims and bones)
struct aiNode *findNode(struct aiNode *node, char *name)
{
	unsigned int i;
	if (!strcmp(name, node->mName.data)) return node;

	for (i = 0; i < node->mNumChildren; i++)
     {
		struct aiNode *found = findNode(node->mChildren[i], name);
		if (found) { return found; }
	 }
	return NULL;
}

// calculate absolute transform for node to do mesh skinning
void transformNodeWithAllParentTransforms(  aiMatrix4x4 *result, struct aiNode *node)
{
	if (node->mParent)
    {
		transformNodeWithAllParentTransforms(result, node->mParent);
		aiMultiplyMatrix4(result, &node->mTransformation);
	} else
	{
		*result = node->mTransformation;
	}
}



void transformMeshOld(struct aiScene *scene , int meshNumber , struct TRI_Model * indexed , struct skeletonHuman * sk )
{
    aiMatrix4x4 skin4;
	unsigned int k;

    struct aiMesh * mesh = scene->mMeshes[meshNumber];
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);

   //Use the parent and node transforms
	for (k = 0; k < mesh->mNumBones; k++)
    {
		struct aiBone *bone = mesh->mBones[k];
		struct aiNode *node = findNode(scene->mRootNode, bone->mName.data);

		aiString boneName = bone->mName;

		fprintf(stderr,"Bone %u - %s  has %u children \n" , k , boneName.C_Str() , node->mNumChildren );

		transformNodeWithAllParentTransforms(&skin4, node);

        modifiedSkeleton.bone[k].nodeTransform = node->mTransformation;
        modifiedSkeleton.bone[k].parentTransform = skin4;

        unsigned int overrideHappened=0;
        unsigned int i=0;

        for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            if (strcmp(boneName.C_Str(),smartBodyNames[i])==0)
              {
		       fprintf(stderr,"hooked with %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" ,jointNames[i] , sk->relativeJointAngle[i].x, sk->relativeJointAngle[i].y, sk->relativeJointAngle[i].z);
               modifiedSkeleton.bone[k].ScalingVec.x=1.0;
               modifiedSkeleton.bone[k].ScalingVec.y=1.0;
               modifiedSkeleton.bone[k].ScalingVec.z=1.0;

               modifiedSkeleton.bone[k].TranslationVec.x=sk->joint[i].x;
               modifiedSkeleton.bone[k].TranslationVec.y=sk->joint[i].y;
               modifiedSkeleton.bone[k].TranslationVec.z=sk->joint[i].z;

              aiMatrix4x4::Scaling(modifiedSkeleton.bone[k].ScalingVec,modifiedSkeleton.bone[k].scalingMat);
              aiMatrix4x4::Translation (modifiedSkeleton.bone[k].TranslationVec,modifiedSkeleton.bone[k].translationMat);
              //aiMakeQuaternion( &modifiedSkeleton.bone[k].rotationMat , &modifiedSkeleton.bone[k].RotationQua );
              modifiedSkeleton.bone[k].rotationMat.FromEulerAnglesXYZ(
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].x),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].y),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].z)
                                                                     );

               modifiedSkeleton.bone[k].nodeTransform   = modifiedSkeleton.bone[k].translationMat  * modifiedSkeleton.bone[k].rotationMat * modifiedSkeleton.bone[k].scalingMat;
               //aiMakeIdentity(&modifiedSkeleton.bone[k].nodeTransform);
               //node->mTransformation = modifiedSkeleton.bone[k].nodeTransform;
               modifiedSkeleton.bone[k].globalTransform = modifiedSkeleton.bone[k].parentTransform * modifiedSkeleton.bone[k].nodeTransform;
               modifiedSkeleton.bone[k].finalTransform  = m_GlobalInverseTransform * modifiedSkeleton.bone[k].globalTransform  * bone->mOffsetMatrix;

               overrideHappened=1;
            }
        }
        if (!overrideHappened)
        {
          //aiMultiplyMatrix4(&skin4, &bone->mOffsetMatrix);
          modifiedSkeleton.bone[k].finalTransform = m_GlobalInverseTransform * modifiedSkeleton.bone[k].parentTransform * bone->mOffsetMatrix;
        }

       skin4 = modifiedSkeleton.bone[k].finalTransform;
       aiPrintMatrix(&skin4);


       //Update all vertices with current weighted transforms for current the current bone
       //stored in skin3 and skin4
       aiMatrix3x3 skin3;
	   extract3x3(&skin3, &skin4);
       for (i = 0; i < bone->mNumWeights; i++)
        {
			unsigned int v = bone->mWeights[i].mVertexId;
			float w = bone->mWeights[i].mWeight;

			aiVector3D position = mesh->mVertices[v];
			aiTransformVecByMatrix4(&position, &skin4);
			indexed->vertices[v*3+0] += position.x * w;
			indexed->vertices[v*3+1] += position.y * w;
			indexed->vertices[v*3+2] += position.z * w;

			aiVector3D normal = mesh->mNormals[v];
			aiTransformVecByMatrix3(&normal, &skin3);
			indexed->normal[v*3+0] += normal.x * w;
			indexed->normal[v*3+1] += normal.y * w;
			indexed->normal[v*3+2] += normal.z * w;
		}
	}
}


void populateInternalRigState(struct aiScene *scene , int meshNumber, struct boneState * bones )
{
  fprintf(stderr,"Populating internal rig state \n" );
  unsigned int i=0,k=0;
  struct aiMesh * mesh = scene->mMeshes[meshNumber];

  modifiedSkeleton.numberOfBones = mesh->mNumBones;

  for (k = 0; k < mesh->mNumBones; k++)
    {
       struct aiBone *bone = mesh->mBones[k];
	   struct aiNode *node = findNode(scene->mRootNode, bone->mName.data);
       snprintf(bones->bone[k].name,128,"%s", bone->mName.data );

       fprintf(stderr,"Bone %u is %s \n" , k ,bones->bone[k].name );

       modifiedSkeleton.bone[k].nodeTransform = node->mTransformation;
       modifiedSkeleton.bone[k].boneBindTransform = bone->mOffsetMatrix;
       modifiedSkeleton.bone[k].itemID=k;

       modifiedSkeleton.bone[k].parentlessNode=1;
       if (node->mParent)
           {
             //fprintf(stderr,"Node %s has a parent \n",node->mName.data);
             unsigned int foundParent = 0;
             struct aiNode * parentNode = node->mParent;
             for (i=0; i<mesh->mNumBones; i++)
                   {
                     struct aiBone *searchBone = mesh->mBones[i];
                     if (strcmp(searchBone->mName.data,parentNode->mName.data )==0)
                     {
                       modifiedSkeleton.bone[k].parentlessNode=0;
                       modifiedSkeleton.bone[k].parentItemID=i;
                       fprintf(stderr,"Parent of %s is %s which has a boneID of %u \n",node->mName.data ,searchBone->mName.data,i);
                       foundParent =1;
                     }
                   }

             if (!foundParent)
             {
               fprintf(stderr,"Could not find parent , parent is marked as root..\n",i);
             }
           } else
           {
             fprintf(stderr,"Node %s has no parent , parent is marked as root..\n",node->mName.data);
           }
    }
}



void gatherAllParentTransformations(aiMatrix4x4 * newParentTransform ,  struct boneState * bones ,  unsigned int k )
{
 aiMatrix4x4 parentChain[30];
 unsigned int nodeIDChain[30]={0};

 fprintf(stderr,"Gathering all parent transformations of node %s : \n",bones->bone[k].name);
 unsigned int z=0,chain=0;
 parentChain[chain] = bones->bone[k].nodeTransform;
 if (!bones->bone[k].parentlessNode)
   {
    z=bones->bone[k].parentItemID;
    while (!bones->bone[z].parentlessNode)
          {
            fprintf(stderr," %u=%s ",chain,bones->bone[z].name);
            ++chain;
            parentChain[chain] = bones->bone[z].nodeTransform;
            nodeIDChain[chain] = z;
            z=bones->bone[z].parentItemID;
          }
     ++chain;
     parentChain[chain] = bones->bone[z].nodeTransform;
     nodeIDChain[chain] = z;
     fprintf(stderr," %u=%s ",chain,bones->bone[z].name);
   }


  if (chain==0)
  {
   fprintf(stderr,"Its a single node\n");
   //*newParentTransform = bones->bone[k].nodeTransform;
   return ;
  } else
  if (chain==1)
  {
   fprintf(stderr,"Its a double node\n");
   *newParentTransform = parentChain[1];
   return ;
  } else
  {
  fprintf(stderr,"And now executing a chain of %u multiplication: \n",chain);
  while (chain>1)
    {
     fprintf(stderr," aiMultiplyMatrix4(%s,%s)\n",bones->bone[nodeIDChain[chain]].name,bones->bone[nodeIDChain[chain-1]].name);

     aiMultiplyMatrix4( &parentChain[chain] , &parentChain[chain-1]);
     parentChain[chain-1]=parentChain[chain];
	 --chain;
    }
  fprintf(stderr,"done\n");
  *newParentTransform = parentChain[2];
  return ;
  }


}


void transformMeshNew(struct aiScene *scene , int meshNumber , struct TRI_Model * indexed , struct skeletonHuman * sk )
{
    aiMatrix4x4 skin4;
	unsigned int k,i,z;

    struct aiMesh * mesh = scene->mMeshes[meshNumber];
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);

    populateInternalRigState(scene , meshNumber, &modifiedSkeleton );

   //Use the parent and node transforms

 #if DO_TRANSFORM
	for (k = 0; k < modifiedSkeleton.numberOfBones; k++)
    {
      //First decide on the node transformations taking place during this operation
       for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            if (strcmp(modifiedSkeleton.bone[k].name , smartBodyNames[i])==0)
              {
               fprintf(stderr,"hooked with %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" ,jointNames[i] , sk->relativeJointAngle[i].x, sk->relativeJointAngle[i].y, sk->relativeJointAngle[i].z);
               modifiedSkeleton.bone[k].ScalingVec.x=1.0;
               modifiedSkeleton.bone[k].ScalingVec.y=1.0;
               modifiedSkeleton.bone[k].ScalingVec.z=1.0;

               //modifiedSkeleton.bone[k].TranslationVec.x=sk->joint[i].x;
               //modifiedSkeleton.bone[k].TranslationVec.y=sk->joint[i].y;
               //modifiedSkeleton.bone[k].TranslationVec.z=sk->joint[i].z;

              aiMatrix4x4::Scaling(modifiedSkeleton.bone[k].ScalingVec,modifiedSkeleton.bone[k].scalingMat);
              aiMatrix4x4::Translation (modifiedSkeleton.bone[k].TranslationVec,modifiedSkeleton.bone[k].translationMat);
              //aiMakeQuaternion( &modifiedSkeleton.bone[k].rotationMat , &modifiedSkeleton.bone[k].RotationQua );
              modifiedSkeleton.bone[k].rotationMat.FromEulerAnglesXYZ(
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].x),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].y),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].z)
                                                                     );

               modifiedSkeleton.bone[k].nodeTransform   = modifiedSkeleton.bone[k].translationMat  * modifiedSkeleton.bone[k].rotationMat * modifiedSkeleton.bone[k].scalingMat;
              }
        }
    }
 #endif // DO_TRANSFORM



	for (k = 0; k < modifiedSkeleton.numberOfBones; k++)
    {
	   struct aiBone *bone = mesh->mBones[k];
	   struct aiNode *node = findNode(scene->mRootNode, bone->mName.data);


       gatherAllParentTransformations(&modifiedSkeleton.bone[k].parentTransform , &modifiedSkeleton , k );
       modifiedSkeleton.bone[k].globalTransform = modifiedSkeleton.bone[k].parentTransform * modifiedSkeleton.bone[k].nodeTransform;
       modifiedSkeleton.bone[k].finalTransform  = m_GlobalInverseTransform * modifiedSkeleton.bone[k].globalTransform  * modifiedSkeleton.bone[k].boneBindTransform;

       //modifiedSkeleton.bone[k].finalTransform = m_GlobalInverseTransform * modifiedSkeleton.bone[k].parentTransform * modifiedSkeleton.bone[k].boneBindTransform;


       //Update all vertices with current weighted transforms for current the current bone
       //stored in skin3 and skin4
       aiMatrix3x3 finalTransform3x3;
	   extract3x3(&finalTransform3x3, &modifiedSkeleton.bone[k].finalTransform);
       for (i = 0; i < bone->mNumWeights; i++)
        {
			unsigned int v = bone->mWeights[i].mVertexId;
			float w = bone->mWeights[i].mWeight;

			aiVector3D position;// = mesh->mVertices[v];
			aiTransformVecByMatrix4(&position, &modifiedSkeleton.bone[k].finalTransform);
			indexed->vertices[v*3+0] += position.x * w;
			indexed->vertices[v*3+1] += position.y * w;
			indexed->vertices[v*3+2] += position.z * w;

			aiVector3D normal;// = mesh->mNormals[v];
			aiTransformVecByMatrix3(&normal, &finalTransform3x3);
			indexed->normal[v*3+0] += normal.x * w;
			indexed->normal[v*3+1] += normal.y * w;
			indexed->normal[v*3+2] += normal.z * w;
		}
	}
}







void transformMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * indexed , struct skeletonHuman * sk )
{
  #if USE_NEW_TRANSFORM_MESH_CODE
    transformMeshNew(scene,meshNumber,indexed,sk);
   #else
    transformMeshOld(scene,meshNumber,indexed,sk);
  #endif // NEWCODE
 return;
}









void prepareMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * triModel )
{
    struct aiMesh * mesh = scene->mMeshes[meshNumber];

    fprintf(stderr,"Preparing mesh %u   \n",meshNumber);
    fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
    fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
    fprintf(stderr,"  %d faces \n",mesh->mNumFaces);
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);

	unsigned int k;
	for (k = 0; k < mesh->mNumBones; k++)
    {
		aiString boneName = mesh->mBones[k]->mName;
		//fprintf(stderr,"   -bone %u - %s \n" , k , boneName.C_Str() );
		//fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[0], mesh->mBones[k]->mOffsetMatrix[1], mesh->mBones[k]->mOffsetMatrix[2], mesh->mBones[k]->mOffsetMatrix[3] );
		//fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[4], mesh->mBones[k]->mOffsetMatrix[5], mesh->mBones[k]->mOffsetMatrix[6], mesh->mBones[k]->mOffsetMatrix[7] );
		//fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[8], mesh->mBones[k]->mOffsetMatrix[9], mesh->mBones[k]->mOffsetMatrix[10],mesh->mBones[k]->mOffsetMatrix[11] );
		//fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[12],mesh->mBones[k]->mOffsetMatrix[13], mesh->mBones[k]->mOffsetMatrix[14], mesh->mBones[k]->mOffsetMatrix[15] );
	}


    fprintf(stderr,"%u color sets",AI_MAX_NUMBER_OF_COLOR_SETS);
    unsigned int colourSet = 0;
        for (colourSet = 0; colourSet< AI_MAX_NUMBER_OF_COLOR_SETS; colourSet++)
        {
          fprintf(stderr," c%u ",colourSet);
        }
    fprintf(stderr," \n");

    fprintf(stderr,"Preparing mesh %u with %u colors \n",meshNumber,mesh->mNumFaces,meshNumber);
	//xxxmesh->texture = loadmaterial(scene->mMaterials[mesh->mMaterialIndex]);

    triModel->header.numberOfVertices      = mesh->mNumVertices*3;
    triModel->header.numberOfNormals       = mesh->mNumVertices*3;
    triModel->header.numberOfTextureCoords = mesh->mNumVertices*2;
    triModel->header.numberOfColors        = mesh->mNumVertices*3;
    triModel->header.numberOfIndices       = mesh->mNumFaces*3;

   //fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);

	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals  * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords     * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors  * sizeof(float));
    triModel->indices        = (unsigned int*) malloc( triModel->header.numberOfIndices  * sizeof(unsigned int));

    unsigned int i=0;

    unsigned int o=0,n=0,t=0,c=0;
	for (i = 0; i < mesh->mNumVertices; i++)
    {
	    triModel->vertices[(i*3)+0] = mesh->mVertices[i].x;
	    triModel->vertices[(i*3)+1] = mesh->mVertices[i].y;
	    triModel->vertices[(i*3)+2] = mesh->mVertices[i].z;
      if (mesh->mNormals)
        {
			triModel->normal[(i*3)+0] = mesh->mNormals[i].x;
			triModel->normal[(i*3)+1] = mesh->mNormals[i].y;
			triModel->normal[(i*3)+2] = mesh->mNormals[i].z;
        }

      if (mesh->mTextureCoords[0])
        {
			triModel->textureCoords[(i*2)+0] = mesh->mTextureCoords[0][i].x;
			triModel->textureCoords[(i*2)+1] = 1 - mesh->mTextureCoords[0][i].y;
		}

        unsigned int colourSet = 0;
        //for (colourSet = 0; colourSet< AI_MAX_NUMBER_OF_COLOR_SETS; colourSet++)
        {
          if(mesh->HasVertexColors(colourSet))
         {
          triModel->colors[(i*3)+0] = mesh->mColors[colourSet][i].r;
          triModel->colors[(i*3)+1] = mesh->mColors[colourSet][i].g;
          triModel->colors[(i*3)+2] = mesh->mColors[colourSet][i].b;
        }
       }

	}

   for (i = 0; i < mesh->mNumFaces; i++)
    {
		struct aiFace *face = mesh->mFaces + i;
		triModel->indices[(i*3)+0] = face->mIndices[0];
		triModel->indices[(i*3)+1] = face->mIndices[1];
		triModel->indices[(i*3)+2] = face->mIndices[2];
    }
}


void deformOriginalModelAndBringBackFlatOneBasedOnThisSkeleton(
                                                                struct TRI_Model * outFlatModel ,
                                                                struct TRI_Model * inOriginalIndexedModel ,
                                                                struct skeletonHuman * sk
                                                              )
{
  struct TRI_Model temporaryIndexedDeformedModel={0};
  fprintf(stderr,"Copying to intermediate mesh\n");
  copyModelTri(&temporaryIndexedDeformedModel,inOriginalIndexedModel);
  fprintf(stderr,"Transforming intermediate mesh\n");
  transformMesh( g_scene , 0 , &temporaryIndexedDeformedModel , sk );
  fprintf(stderr,"Flattening intermediate mesh\n");
  fillFlatModelTriFromIndexedModelTri(outFlatModel , &temporaryIndexedDeformedModel);

  fprintf(stderr,"Deallocating intermediate mesh\n");
  deallocModelTri(&temporaryIndexedDeformedModel);
  fprintf(stderr,"Serving back flattened mesh\n");
}




void prepareScene(struct aiScene *scene , struct TRI_Model * triModel , struct TRI_Model * originalModel)
{
    fprintf(stderr,"Preparing scene with %u meshes\n",scene->mNumMeshes);
    if (scene->mNumMeshes>1) { fprintf(stderr,"Can only handle single meshes atm \n"); }

       fprintf(stderr,"Reading mesh from collada \n");
       prepareMesh(scene, 0,  originalModel );


/*
 struct skeletonHuman sk={0};
  unsigned int newVal = 300;// rand()%330;
  sk.relativeJointAngle[HUMAN_SKELETON_LEFT_KNEE].x=newVal;
  sk.relativeJointAngle[HUMAN_SKELETON_LEFT_KNEE].y=newVal;

  sk.relativeJointAngle[HUMAN_SKELETON_RIGHT_KNEE].x=newVal;
  sk.relativeJointAngle[HUMAN_SKELETON_RIGHT_KNEE].y=newVal;
 transMesh( g_scene , 0 , originalModel , &sk );
*/

       fprintf(stderr,"Flattening mesh\n");
       fillFlatModelTriFromIndexedModelTri(triModel , originalModel);

    return ;

	unsigned int i=0;
	for (i = 0; i < scene->mNumMeshes; i++)
    {
		prepareMesh(scene, i,  triModel );
		//TODO HANDLE MORE THAN ONE MESHES
		//transformmesh(scene, meshlist + i);
	}

}



int testAssimp(const char * filename  , struct TRI_Model * triModel , struct TRI_Model * originalModel)
{
int flags = aiProcess_Triangulate;
		flags |= aiProcess_JoinIdenticalVertices;
		flags |= aiProcess_GenSmoothNormals;
		flags |= aiProcess_GenUVCoords;
		flags |= aiProcess_TransformUVCoords;
		flags |= aiProcess_RemoveComponent;

		g_scene = (struct aiScene*) aiImportFile(  filename, flags);
		if (g_scene)
        {
            m_GlobalInverseTransform = g_scene->mRootNode->mTransformation;
            m_GlobalInverseTransform.Inverse();

            prepareScene(g_scene,triModel,originalModel);

            return 1;
		} else
		{
			fprintf(stderr, "cannot import scene: '%s'\n", filename);
		}

  return 0;
}
