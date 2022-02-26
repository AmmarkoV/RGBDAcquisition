#include "model_loader_assimp.h"

#include <stdio.h>
#include <stdlib.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


//#include "../../../../tools/Primitives/skeleton.h"

#define DO_TRANSFORM 1
#define USE_NEW_TRANSFORM_MESH_CODE 1



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


static float defaultJointsOffsetXYZ[] = {
                                      0 , 0 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , 20 ,
                                      0 , 45 , 0 ,
                                      0 , 45 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 90 , -90 ,
                                      0 , 90 , 90 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , 0 ,
                                      0 , 0 , -45 ,
                                      0 , 0 , -45 ,
                                      0 , -90 , -90
                                     };

static float defaultJointsOffsetZXY[] = {
                                         0 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         0 , 20 , 0 ,
                                         45 , 0, 0 ,
                                         45 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         90 , -90 , 0 ,
                                         90 , 90 , 0 ,
                                         0 , 0 , 0 ,
                                         0 , 0 , 0 ,
                                         0 ,  -45 , 0 ,
                                         0 ,  -45 , 0 ,
                                         -90 , -90 , 0
                                        };




/*

void readNodeHierarchyNEW(const aiMesh * mesh , const aiNode* pNode,  struct boneState * bones , struct skeletonHuman * sk, aiMatrix4x4 & ParentTransform , unsigned int recursionLevel)
{
    //Print out node tree with space seperated readable form
    if (recursionLevel==0)    { fprintf(stderr,"readNodeHierarchy : \n"); } else
                              { fprintf(stderr,"   "); }
    fprintf(stderr,"%s\n" , pNode->mName.data);


    //The default Node Transformation is no transformation at all , so if we don't have something to change
    //NodeTransformation is going to have the default value
    aiMatrix4x4 NodeTransformation=pNode->mTransformation;


    //Assimp has two structures to facilitate rigged meshes , aiNodes and AiBones , they have their own indexing systems
    //the next call will try to find the bone that is associated with the current pNode to store the resulting matrices
    //at the correct place
    unsigned int i=0 , foundBone=0 , usedBone=0;
    unsigned int boneNumber=findBoneFromString(mesh,pNode->mName.data,&foundBone);

    //If we found a bone then we can intervene because we have an inverse bind matrix
    if  (foundBone)
    {
      //We search our internal (skeleton.h declared ) structure to find if we have some new data on the specific joint
      for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          //In case there is a node in skeleton.h that corresponds to the specific aiBone we can try to check if it is currently active in our target skeleton model
            if (strcmp(pNode->mName.data , smartBodyNames[i])==0)
              {
               //If we have a target in our skeleton model , we need to calculate the new Node Transformation
               if ( sk->active[i] )
               {
               fprintf(stderr,GREEN "hooked with %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" NORMAL ,jointNames[i] , sk->relativeJointAngle[i].x, sk->relativeJointAngle[i].y, sk->relativeJointAngle[i].z);
               bones->bone[boneNumber].ScalingVec.x=1.0;
               bones->bone[boneNumber].ScalingVec.y=1.0;
               bones->bone[boneNumber].ScalingVec.z=1.0;

               bones->bone[boneNumber].TranslationVec.x=pNode->mTransformation.a4;
               bones->bone[boneNumber].TranslationVec.y=pNode->mTransformation.b4;
               bones->bone[boneNumber].TranslationVec.z=pNode->mTransformation.c4;

               aiMatrix4x4::Scaling(bones->bone[boneNumber].ScalingVec,bones->bone[boneNumber].scalingMat);
               aiMatrix4x4::Translation (bones->bone[boneNumber].TranslationVec,bones->bone[boneNumber].translationMat);
               //aiMakeQuaternion( &bones->bone[boneNumber].rotationMat , &bones.bone[k].RotationQua );
               //aiPrintMatrix(&bones->bone[boneNumber].rotationMat );

               bones->bone[boneNumber].rotationMat.FromEulerAnglesXYZ(
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].x + defaultJointsOffset[i*3+0] ),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].y + defaultJointsOffset[i*3+1] ),
                                                                      degrees_to_rad ( sk->relativeJointAngle[i].z + defaultJointsOffset[i*3+2] )
                                                                     );

               NodeTransformation =  bones->bone[boneNumber].translationMat * bones->bone[boneNumber].rotationMat * bones->bone[boneNumber].scalingMat;
               usedBone=1;
              } else
              {
               fprintf(stderr,YELLOW " inactive %s ( r.x=%0.2f r.y=%0.2f r.z=%0.2f ) !\n" NORMAL,jointNames[i] , sk->relativeJointAngle[i].x, sk->relativeJointAngle[i].y, sk->relativeJointAngle[i].z);
              }
            }
        }
    } else
    {
      fprintf(stderr,RED "        <!%s!>\n" NORMAL,pNode->mName.data);
    }

    aiMatrix4x4 GlobalTransformation = ParentTransform  * NodeTransformation;
    if (!usedBone)
        {
          fprintf(stderr," did not use bone %u , %s\n", boneNumber , mesh->mBones[boneNumber]->mName.data);
        } else
        {
          //we used bone , update its final transform
        }

    bones->bone[boneNumber].finalTransform = m_GlobalInverseTransform * GlobalTransformation * bones->bone[boneNumber].boneInverseBindTransform;
    for ( i = 0 ; i < pNode->mNumChildren ; i++)
    {
        readNodeHierarchyNEW(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
    }
}

*/



struct boneItem
{
  aiMatrix4x4 finalTransform;
  aiMatrix4x4 parentTransform;
  aiMatrix4x4 nodeTransform;
  aiMatrix4x4 nodeTransformInitial;
  aiMatrix4x4 globalTransform;
  aiMatrix4x4 boneInverseBindTransform;

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
  struct boneItem bone[200];
  unsigned int numberOfBones;

};

struct aiScene *g_scene = NULL;
aiMatrix4x4 m_GlobalInverseTransform;

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

// find a node by name in the hierarchy (for anims and bones)
unsigned int findBoneFromString(const aiMesh * mesh, const char *name , unsigned int * foundBone)
{
    *foundBone = 0;
	unsigned int k;
	for (k = 0; k < mesh->mNumBones; k++)
    {
		struct aiBone *bone = mesh->mBones[k];
         if (strcmp(bone->mName.data,name)==0)
              {
                   *foundBone = 1;
	               return k;
              }
    }
	return 0;
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

void populateInternalRigState(struct aiScene *scene , int meshNumber, struct boneState * bones )
{
  //fprintf(stderr,"Populating internal rig state \n" );
  unsigned int i=0 , k=0;
  struct aiMesh * mesh = scene->mMeshes[meshNumber];

  bones->numberOfBones = mesh->mNumBones;

  for (k = 0; k < mesh->mNumBones; k++)
    {
       struct aiBone *bone = mesh->mBones[k];
	   struct aiNode *node = findNode(scene->mRootNode, bone->mName.data);
       snprintf(bones->bone[k].name,128,"%s", bone->mName.data );

       //fprintf(stderr,"Bone %u is %s \n" , k ,bones->bone[k].name );

       bones->bone[k].nodeTransform = node->mTransformation;
       bones->bone[k].nodeTransformInitial = node->mTransformation;
       bones->bone[k].boneInverseBindTransform = bone->mOffsetMatrix;
       bones->bone[k].itemID=k;

       bones->bone[k].parentlessNode=1;

       //Decide if the node is parentless
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
                       bones->bone[k].parentlessNode=0;
                       bones->bone[k].parentItemID=i;
                       //fprintf(stderr,"Parent of %s is %s which has a boneID of %u \n",node->mName.data ,searchBone->mName.data,i);
                       foundParent =1;
                     }
                   }

             if (!foundParent)
             {
               //fprintf(stderr,"Could not find parent , parent is marked as root..\n");
             }
           } else
           {
             //fprintf(stderr,"Node %s has no parent , parent is marked as root..\n",node->mName.data);
           }
    }
}

/*
void readNodeHierarchyOLD(const aiMesh * mesh , const aiNode* pNode,  struct boneState * bones , struct skeletonHuman * sk, aiMatrix4x4 & ParentTransform , unsigned int recursionLevel)
{

    if (recursionLevel==0)    { fprintf(stderr,"readNodeHierarchy : \n"); } else
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


//               bones->bone[boneNumber].rotationMat.FromEulerAnglesXYZ(
//                                                                      degrees_to_rad ( sk->relativeJointAngle[i].x + defaultJointsOffsetXYZ[i*3+0] ),
//                                                                      degrees_to_rad ( sk->relativeJointAngle[i].y + defaultJointsOffsetXYZ[i*3+1] ),
//                                                                      degrees_to_rad ( sk->relativeJointAngle[i].z + defaultJointsOffsetXYZ[i*3+2] )
//                                                                      );

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
        readNodeHierarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
    }
    } else
    {
      aiMatrix4x4 GlobalTransformation = ParentTransform  * pNode->mTransformation;
      fprintf(stderr,"        <!%s!>\n",pNode->mName.data);
       for ( i = 0 ; i < pNode->mNumChildren ; i++)
       {
         readNodeHierarchyOLD(mesh,pNode->mChildren[i],bones,sk,GlobalTransformation,recursionLevel+1);
       }
    }
}
*/


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
    //readNodeHierarchyOLD(mesh,scene->mRootNode,&modifiedSkeleton,sk,Identity,0);

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

void prepareMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * triModel )
{
    struct aiMesh * mesh = scene->mMeshes[meshNumber];

    fprintf(stderr,"Preparing mesh %u   \n",meshNumber);
    fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
    fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
    fprintf(stderr,"  %d faces \n",mesh->mNumFaces);
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);



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
  //visualize2DSkeletonHuman("deformOriginalModelAndBringBackFlatOneBasedOnThisSkeleton.svg",sk,3.0);

  struct TRI_Model temporaryIndexedDeformedModel={0};
  fprintf(stderr,"Copying to intermediate mesh\n");
  //copyModelTri(&temporaryIndexedDeformedModel,inOriginalIndexedModel);
  fprintf(stderr,"Transforming intermediate mesh\n");
  transformMeshBasedOnSkeleton( g_scene , 0 , &temporaryIndexedDeformedModel , sk );
  fprintf(stderr,"Flattening intermediate mesh\n");
  fillFlatModelTriFromIndexedModelTri(outFlatModel , &temporaryIndexedDeformedModel);

  fprintf(stderr,"Deallocating intermediate mesh\n");
  //deallocModelTri(&temporaryIndexedDeformedModel);
  fprintf(stderr,"Serving back flattened mesh\n");
}

void prepareScene(struct aiScene *scene , struct TRI_Model * triModel , struct TRI_Model * originalModel)
{
    fprintf(stderr,"Preparing scene with %u meshes\n",scene->mNumMeshes);
    if (scene->mNumMeshes>1) { fprintf(stderr,"Can only handle single meshes atm \n"); }

       fprintf(stderr,"Reading mesh from collada \n");
       prepareMesh(scene, 0,  originalModel );

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
