#include "assimp_loader.h"

#include <stdio.h>
#include <stdlib.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


struct aiScene *g_scene = NULL;
aiMatrix4x4 m_GlobalInverseTransform;

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


void floatMatMakeIdentity(float * am)
{
 am[0] = 1.0; am[1] = 0.0;  am[2] = 0.0;  am[3] = 0.0;
 am[4] = 0.0; am[5] = 1.0;  am[6] = 0.0;  am[7] = 0.0;
 am[8] = 0.0; am[9] = 0.0;  am[10]= 1.0;  am[11]= 0.0;
 am[12]= 0.0; am[13]= 0.0;  am[14]= 0.0;  am[15]= 1.0;
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
 similarity += (am->a1==bm->a1); similarity += (am->a2==bm->a2); similarity += (am->a3==bm->a3); similarity += (am->a4==bm->a4);
 similarity += (am->b1==bm->b1); similarity += (am->b2==bm->b2); similarity += (am->b3==bm->b3); similarity += (am->b4==bm->b4);
 similarity += (am->c1==bm->c1); similarity += (am->c2==bm->c2); similarity += (am->c3==bm->c3); similarity += (am->c4==bm->c4);
 similarity += (am->d1==bm->d1); similarity += (am->d2==bm->d2); similarity += (am->d3==bm->d3); similarity += (am->d4==bm->d4);

 return (similarity==16);
}

void aiPrintMatrix(aiMatrix4x4 * am)
{
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->a1, am->a2,  am->a3,   am->a4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->b1, am->b2,  am->b3,   am->b4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->c1, am->c2,  am->c3,   am->c4 );
 fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , am->d1, am->d2,  am->d3,   am->d4 );
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

int findBoneNumFromAINode(const aiNode * pNode , struct aiMesh * mesh , unsigned int * boneNum)
{
    unsigned int k=0;
  for (k = 0; k < mesh->mNumBones; k++)
    {
       struct aiBone *bone = mesh->mBones[k];
       if (strcmp(pNode->mName.data , bone->mName.data)==0)
              {
                *boneNum = k;
                return 1;
              }

    }
 return 0;
}


void countNumberOfNodesInternal(struct aiNode *node , unsigned int * numberOfNodes)
{
  (*numberOfNodes)++;

  unsigned int i=0;
   for ( i = 0 ; i < node->mNumChildren ; i++)
        {
          countNumberOfNodesInternal(node->mChildren[i],numberOfNodes);
        }
}

unsigned int countNumberOfNodes(struct aiScene *scene  , struct aiMesh * mesh )
{
  unsigned int numberOfNodes = 0;
  countNumberOfNodesInternal(scene->mRootNode , &numberOfNodes);
  return numberOfNodes;
}

void convertMatrixAIToAmMatrix(float * rm ,  aiMatrix4x4 * am)
{
 rm[0]=am->a1; rm[1]=am->a2;  rm[2]=am->a3;  rm[3]=am->a4;
 rm[4]=am->b1; rm[5]=am->b2;  rm[6]=am->b3;  rm[7]=am->b4;
 rm[8]=am->c1; rm[9]=am->c2;  rm[10]=am->c3; rm[11]=am->c4;
 rm[12]=am->d1;rm[13]=am->d2; rm[14]=am->d3; rm[15]=am->d4;
}


void fillInNodeAndBoneData(struct aiNode *node ,  struct aiMesh * mesh , unsigned int * numberOfNodes , unsigned int parentNodeID  , struct TRI_Model * triModel)
{
  if ( triModel->bones == 0 ) { fprintf(stderr,"Can't fill node info in empty bone structure \n"); return; }
  struct aiBone *bone = 0;
  unsigned int nodeNum = *numberOfNodes;
  unsigned int boneNum = 0 , bufSize=0;


  triModel->bones[nodeNum].info = (struct TRI_Bones_Header * )  malloc ( sizeof (struct TRI_Bones_Header) );
  if (!triModel->bones[nodeNum].info) { fprintf(stderr,"Can't allocate enough space for bone info \n"); return; }
  memset(triModel->bones[nodeNum].info , 0 , sizeof (struct TRI_Bones_Header));


  //Make sure that the finalVertexTransformation is identity , so it will always make sense..
  float * d = triModel->bones[nodeNum].info->finalVertexTransformation;
  d[0]=1.0;  d[1]=0.0;  d[2]=0.0;  d[3]=0.0;
  d[4]=0.0;  d[5]=1.0;  d[6]=0.0;  d[7]=0.0;
  d[8]=0.0;  d[9]=0.0;  d[10]=1.0; d[11]=0.0;
  d[12]=0.0; d[13]=0.0; d[14]=0.0; d[15]=1.0;


  if (parentNodeID==nodeNum)
  {
    //We are the root node so we won't update in our parents children
  } else
  {
    if (triModel->bones[parentNodeID].boneChild == 0 )
    {
      fprintf(stderr,RED "BUG : Our parent %s(%u) does not have any space allocated (%p) to store his children %s(%u)..!\n" NORMAL,
                      triModel->bones[parentNodeID].boneName ,
                      parentNodeID ,
                      triModel->bones[parentNodeID].boneChild ,
                      node->mName.data ,
                      nodeNum
             );
    } else
    if (triModel->bones[parentNodeID].info->allocatedNumberOfBoneChildren <= triModel->bones[parentNodeID].info->numberOfBoneChildren )
    {
      fprintf(stderr,RED "BUG : Our parent did not allocate enough space for all his children , what a terrible parent..! \n" NORMAL);
    } else
    {
     //We are a child of our parent so let's store that..!
     unsigned int existingChildren = triModel->bones[parentNodeID].info->numberOfBoneChildren;

     //fprintf(stderr,"Node %u now has %u/%u children ( %u is a new one of them ) ..! \n ",parentNodeID , existingChildren+1 , triModel->bones[parentNodeID].info->allocatedNumberOfBoneChildren , nodeNum);
     triModel->bones[parentNodeID].boneChild[existingChildren] = nodeNum;
     ++triModel->bones[parentNodeID].info->numberOfBoneChildren;
    }
  }


  triModel->bones[nodeNum].info->altered = 0; //Originally unaltered mesh
  triModel->bones[nodeNum].info->boneParent = parentNodeID;

  //We pass the node name
  triModel->bones[nodeNum].info->boneNameSize = strlen(node->mName.data);
  triModel->bones[nodeNum].boneName = (char* ) malloc(sizeof(char) * (1+triModel->bones[nodeNum].info->boneNameSize) );
  if (triModel->bones[nodeNum].boneName)
        { memcpy(triModel->bones[nodeNum].boneName,node->mName.data ,triModel->bones[nodeNum].info->boneNameSize); }
  triModel->bones[nodeNum].boneName[triModel->bones[nodeNum].info->boneNameSize]=0;


   convertMatrixAIToAmMatrix(triModel->bones[nodeNum].info->localTransformation,  &node->mTransformation);
   floatMatMakeIdentity(triModel->bones[nodeNum].info->finalVertexTransformation);
   floatMatMakeIdentity(triModel->bones[nodeNum].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose);


  if (findBoneNumFromAINode( node ,  mesh , &boneNum ))
    {
      unsigned int k=0;
      bone = mesh->mBones[boneNum];
      convertMatrixAIToAmMatrix(triModel->bones[nodeNum].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose,  &bone->mOffsetMatrix);
      //We have an associated bone structure with our node ! :)
      triModel->bones[nodeNum].info->boneWeightsNumber = bone->mNumWeights;

      bufSize = sizeof(float) * triModel->bones[nodeNum].info->boneWeightsNumber;
      triModel->bones[nodeNum].weightValue = (float*)        malloc(bufSize);
      if (triModel->bones[nodeNum].weightValue)
       {
         for (k = 0; k < triModel->bones[nodeNum].info->boneWeightsNumber; k++)
         {
           triModel->bones[nodeNum].weightValue[k] = bone->mWeights[k].mWeight;
         }
       }

       bufSize = sizeof(unsigned int) * triModel->bones[nodeNum].info->boneWeightsNumber;
       triModel->bones[nodeNum].weightIndex = (unsigned int*) malloc(bufSize);

       if (triModel->bones[nodeNum].weightIndex)
       {
        for (k = 0; k < triModel->bones[nodeNum].info->boneWeightsNumber; k++)
         {
           triModel->bones[nodeNum].weightIndex[k] = bone->mWeights[k].mVertexId;
	     }
       }
    } else
    {
      //No bone structure no weights no nothing..!
      triModel->bones[nodeNum].info->boneWeightsNumber=0;
      triModel->bones[nodeNum].weightValue=0;
      triModel->bones[nodeNum].weightIndex=0;
    }



  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //We are done processing this node , going to the next one now..!
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  parentNodeID=nodeNum; //We are the parent now..!
  (*numberOfNodes)++;   //We are going to the next node..!

  //fprintf(stderr,"Node has %u children , including them ",node->mNumChildren);
  triModel->bones[nodeNum].info->allocatedNumberOfBoneChildren = node->mNumChildren;
  triModel->bones[nodeNum].info->numberOfBoneChildren = 0;
  triModel->bones[nodeNum].boneChild = 0;
  if (node->mNumChildren>0)
  { //If we have children then we need to allocate enough space for them to fill in their selves
  //fprintf(stderr,"Node %s(%u) is a responsible parent with %u children allocating enough space for them.. \n",triModel->bones[nodeNum].boneName,nodeNum,node->mNumChildren);
  triModel->bones[nodeNum].boneChild = (unsigned int *) malloc (  sizeof(unsigned int) * (node->mNumChildren+1) );

  //fprintf(stderr,"Allocated @ %p  \n",triModel->bones[nodeNum].info->boneChild);

   for (unsigned int i = 0 ; i < node->mNumChildren ; i++)
        {
          fillInNodeAndBoneData(node->mChildren[i] ,  mesh , numberOfNodes , parentNodeID  , triModel);
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

    fprintf(stderr,"Preparing mesh %u with %u colors \n",meshNumber,mesh->mNumFaces);
    if(!mesh->HasVertexColors(colourSet))
         {
          fprintf(stderr,"  Mesh has no vertex colors..!\n");
         }
	//xxxmesh->texture = loadmaterial(scene->mMaterials[mesh->mMaterialIndex]);

    unsigned int verticesSize,normalsSize,textureCoordsSize,colorSize,indexSize;

    triModel->header.triType = TRI_LOADER_VERSION;
    triModel->header.floatSize = sizeof(float);
    triModel->header.TRIMagic[0] = 'T';
    triModel->header.TRIMagic[1] = 'R';
    triModel->header.TRIMagic[2] = 'I';
    triModel->header.TRIMagic[3] = '3';
    triModel->header.TRIMagic[4] = 'D';

    triModel->header.numberOfVertices      = mesh->mNumVertices*3;    verticesSize     =triModel->header.numberOfVertices      * sizeof(float);
    triModel->header.numberOfNormals       = mesh->mNumVertices*3;    normalsSize      =triModel->header.numberOfNormals       * sizeof(float);
    triModel->header.numberOfTextureCoords = mesh->mNumVertices*2;    textureCoordsSize=triModel->header.numberOfTextureCoords * sizeof(float);
    triModel->header.numberOfColors        = mesh->mNumVertices*3;    colorSize        =triModel->header.numberOfColors        * sizeof(float);
    triModel->header.numberOfIndices       = mesh->mNumFaces*3;       indexSize        =triModel->header.numberOfIndices       * sizeof(unsigned int);
    triModel->header.numberOfBones         = 0;                       //bonesSize        =0; //Initially no bones allocated this will be done later..!

    float * gm = triModel->header.boneGlobalInverseTransform;
    gm[0]=m_GlobalInverseTransform.a1;  gm[1]=m_GlobalInverseTransform.a2;  gm[2]=m_GlobalInverseTransform.a3;  gm[3]=m_GlobalInverseTransform.a4;
    gm[4]=m_GlobalInverseTransform.b1;  gm[5]=m_GlobalInverseTransform.b2;  gm[6]=m_GlobalInverseTransform.b3;  gm[7]=m_GlobalInverseTransform.b4;
    gm[8]=m_GlobalInverseTransform.c1;  gm[9]=m_GlobalInverseTransform.c2;  gm[10]=m_GlobalInverseTransform.c3; gm[11]=m_GlobalInverseTransform.c4;
    gm[12]=m_GlobalInverseTransform.d1; gm[13]=m_GlobalInverseTransform.d2; gm[14]=m_GlobalInverseTransform.d3; gm[15]=m_GlobalInverseTransform.d4;

   //fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);
	triModel->vertices      = (float*)            malloc( verticesSize );
	triModel->normal        = (float*)            malloc( normalsSize );
	triModel->textureCoords = (float*)            malloc( textureCoordsSize );
    triModel->colors        = (float*)            malloc( colorSize );
    triModel->indices       = (unsigned int*)     malloc( indexSize );

    fprintf(stderr,"Model has %u UV Channels\n",mesh->GetNumUVChannels());
    fprintf(stderr,"Allocating :   \n");
    fprintf(stderr,"  %u bytes of vertices \n",verticesSize);
    fprintf(stderr,"  %u bytes of normals \n",normalsSize);
    fprintf(stderr,"  %d bytes of textureCoords \n",textureCoordsSize);
    fprintf(stderr,"  %d bytes of colors\n",colorSize);
    fprintf(stderr,"  %d bytes of indices\n",indexSize);

    memset(triModel->vertices,      0 , verticesSize );
    memset(triModel->normal,        0 , normalsSize );
    memset(triModel->textureCoords, 0 , textureCoordsSize );
    memset(triModel->colors,        0 , colorSize );
    memset(triModel->indices,       0 , indexSize );


    #define DO_COLOR_RANGE_CHECK 0

    #if DO_COLOR_RANGE_CHECK
    //-----------------------------------------------------------------------------------------
    fprintf(stderr,"Checking color range.. ");
    float maximum=0;
    for (i = 0; i < mesh->mNumVertices; i++)
    {
      if(mesh->HasVertexColors(colourSet))
         {
          if (maximum<mesh->mColors[colourSet][i].r) { maximum=mesh->mColors[colourSet][i].r; }
          if (maximum<mesh->mColors[colourSet][i].r) { maximum=mesh->mColors[colourSet][i].g; }
          if (maximum<mesh->mColors[colourSet][i].r) { maximum=mesh->mColors[colourSet][i].b; }
         }
    }
    fprintf(stderr," 0.00 - %0.2f\n",maximum);
    if (maximum==0.0)
    {
      fprintf(stderr," Color provided is black everywhere..\n");
    }
    //-----------------------------------------------------------------------------------------
    #endif // DO_COLOR_RANGE_CHECK

	for (unsigned int vertexID = 0; vertexID < mesh->mNumVertices; vertexID++)
    {
	  triModel->vertices[(vertexID*3)+0] = mesh->mVertices[vertexID].x;
	  triModel->vertices[(vertexID*3)+1] = mesh->mVertices[vertexID].y;
	  triModel->vertices[(vertexID*3)+2] = mesh->mVertices[vertexID].z;

      if (mesh->mNormals)
        {
		 triModel->normal[(vertexID*3)+0] = mesh->mNormals[vertexID].x;
		 triModel->normal[(vertexID*3)+1] = mesh->mNormals[vertexID].y;
		 triModel->normal[(vertexID*3)+2] = mesh->mNormals[vertexID].z;
        }

      for (unsigned int uvChannel=0; uvChannel<mesh->GetNumUVChannels(); uvChannel++)
      {
       if (mesh->mTextureCoords[uvChannel])
        {
		 triModel->textureCoords[(vertexID*2)+0] = mesh->mTextureCoords[uvChannel][vertexID].x;
		 triModel->textureCoords[(vertexID*2)+1] = mesh->mTextureCoords[uvChannel][vertexID].y; // aiProcess_FlipUVs does the flip here now .. | 1 -  y
		}
      }

       unsigned int colourSet = 0;
       //for (colourSet = 0; colourSet< AI_MAX_NUMBER_OF_COLOR_SETS; colourSet++)
        {
         if(mesh->HasVertexColors(colourSet))
         {
          triModel->colors[(vertexID*3)+0] = mesh->mColors[colourSet][vertexID].r;
          triModel->colors[(vertexID*3)+1] = mesh->mColors[colourSet][vertexID].g;
          triModel->colors[(vertexID*3)+2] = mesh->mColors[colourSet][vertexID].b;
         }
       }
	}

/*   for(i = 0; i < face->mNumIndices; i++)		// go through all vertices in face
			{
				int vertexIndex = face->mIndices[i];	// get group index for current index
				if(mesh->mColors[0] != nullptr)
					Color4f(&mesh->mColors[0][vertexIndex]);
				if(mesh->mNormals != nullptr)

					if(mesh->HasTextureCoords(0))		//HasTextureCoords(texture_coordinates_set)
					{
						glTexCoord2f(mesh->mTextureCoords[0][vertexIndex].x, 1 - mesh->mTextureCoords[0][vertexIndex].y); //mTextureCoords[channel][vertex]
					}

					glNormal3fv(&mesh->mNormals[vertexIndex].x);
					glVertex3fv(&mesh->mVertices[vertexIndex].x);
			}*/

   for (unsigned int faceID = 0; faceID < mesh->mNumFaces; faceID++)
    {
		struct aiFace *face = mesh->mFaces + faceID;

        if (face->mNumIndices==3)
		 {
		  triModel->indices[(faceID*3)+0] = face->mIndices[0];
		  triModel->indices[(faceID*3)+1] = face->mIndices[1];
		  triModel->indices[(faceID*3)+2] = face->mIndices[2];
         } else
         {
            fprintf(stderr," \n\n\n\n\n Non triangulated face %u \n\n\n\n\n",face->mNumIndices);
            return;
         }
    }

   triModel->header.rootBone=0; //Initial bone is always the root bone..!
   triModel->header.numberOfBones         = countNumberOfNodes( scene  , mesh );
   unsigned int bonesSize                 =triModel->header.numberOfBones         * sizeof(struct TRI_Bones);
   fprintf(stderr,"  %d bytes of bones ( %u nodes ) \n",bonesSize , triModel->header.numberOfBones);


    triModel->bones         = (struct TRI_Bones*) malloc( bonesSize );
    if (triModel->bones!=0)
    {
      memset(triModel->bones, 0 , bonesSize );
      unsigned int numberOfNodesFilled = 0;
      fillInNodeAndBoneData(scene->mRootNode ,  mesh , &numberOfNodesFilled , 0 , triModel);
      fprintf(stderr,"Doing printout of final bone structure as stored.. \n");
      printTRIBoneStructure(triModel, 0);
    } else
    {
      fprintf(stderr,"Could not allocate enough space for bones.. \n");
    }
}



void prepareScene(struct aiScene *scene , struct TRI_Model * triModel , struct TRI_Model * originalModel , int returnIndexedModel , int selectMesh)
{
    fprintf(stderr,"Preparing scene with %u meshes\n",scene->mNumMeshes);
    if (scene->mNumMeshes>1)
     {
     fprintf(stderr,"Can only handle single meshes atm \n");

	 unsigned int i=0;
	 for (i = 0; i < scene->mNumMeshes; i++)
     {
      struct aiMesh * mesh = scene->mMeshes[i];
      fprintf(stderr,"Mesh #%u (%s)   \n",i , mesh->mName.data);
      fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
      fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
      fprintf(stderr,"  %d faces \n",mesh->mNumFaces);
      fprintf(stderr,"  %d or %d bones\n",mesh->mNumBones,countNumberOfNodes(scene,mesh));
     }
    }

       fprintf(stderr,"Reading mesh from collada \n");
       prepareMesh(scene, selectMesh ,  originalModel );

       if (returnIndexedModel)
       {
        fprintf(stderr,"Giving back indexed mesh\n");
        tri_copyModel( triModel , originalModel , 1, 1);
       } else
       {
        fprintf(stderr,"Flattening mesh\n");
        fillFlatModelTriFromIndexedModelTri(triModel , originalModel);
       }

    return ;

	unsigned int i=0;
	for (i = 0; i < scene->mNumMeshes; i++)
    {
		prepareMesh(scene, i,  triModel );
		//TODO HANDLE MORE THAN ONE MESHES
		//transformmesh(scene, meshlist + i);
	}
}



int prepareTRIContainerScene(struct aiScene *scene , struct TRI_Container * triContainer)
{
  triContainer->header.triType = TRI_LOADER_VERSION;
  triContainer->header.floatSize = sizeof(float);
  triContainer->header.TRIMagic[0] = 'T';
  triContainer->header.TRIMagic[1] = 'R';
  triContainer->header.TRIMagic[2] = 'I';
  triContainer->header.TRIMagic[3] = 'C';
  triContainer->header.TRIMagic[4] = 'O';
  //---------------------------------------------
  triContainer->header.floatSize = sizeof(float);
  //---------------------------------------------
  triContainer->header.notUsed1 = 0;
  triContainer->header.notUsed2 = 0;
  triContainer->header.notUsed3 = 0;
  triContainer->header.notUsed4 = 0;
  triContainer->header.notUsed5 = 0;

  triContainer->header.numberOfMeshes = scene->mNumMeshes;
  triContainer->mesh = (struct TRI_Model *) malloc(sizeof(struct TRI_Model) * triContainer->header.numberOfMeshes);

  if (triContainer->mesh!=0)
  {
   for (unsigned int selectedMesh=0; selectedMesh<triContainer->header.numberOfMeshes; selectedMesh++)
     {
       struct aiMesh * mesh = scene->mMeshes[selectedMesh];
       fprintf(stderr,"Mesh #%u (%s)   \n",selectedMesh , mesh->mName.data);
       fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
       fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
       fprintf(stderr,"  %d faces \n",mesh->mNumFaces);
       fprintf(stderr,"  %d or %d bones\n",mesh->mNumBones,countNumberOfNodes(scene,mesh));

       fprintf(stderr,"Reading mesh from collada \n");
       prepareMesh(scene,selectedMesh,&triContainer->mesh[selectedMesh] );

       triContainer->mesh[selectedMesh].header.nameSize = strlen(mesh->mName.data);
       triContainer->mesh[selectedMesh].name = (char * ) malloc( (triContainer->mesh[selectedMesh].header.nameSize+1) * sizeof(char) );
       if (triContainer->mesh[selectedMesh].name!=0)
          { snprintf(triContainer->mesh[selectedMesh].name,triContainer->mesh[selectedMesh].header.nameSize,"%s",mesh->mName.data); }
     }

    return 1;
   }


  return 0;
}




int convertAssimpToTRIContainer(const char * filename  , struct TRI_Container * triContainer)
{
    int flags = aiProcess_Triangulate;
		flags |= aiProcess_JoinIdenticalVertices;
		flags |= aiProcess_GenSmoothNormals;
		flags |= aiProcess_GenUVCoords;
		flags |= aiProcess_TransformUVCoords;
		flags |= aiProcess_RemoveComponent;

		g_scene = (struct aiScene*) aiImportFile( filename, flags);
		if (g_scene)
        {
            //m_GlobalInverseTransform = g_scene->mRootNode->mTransformation;
            //m_GlobalInverseTransform.Inverse();

            prepareTRIContainerScene(g_scene,triContainer);

            aiReleaseImport(g_scene);
            return 1;
		} else
		{
			fprintf(stderr, "Assimp Cannot import scene: '%s'\n", filename);
			fprintf(stderr, " error '%s'\n", aiGetErrorString());
		}

  return 0;
}





int convertAssimpToTRI(const char * filename  , struct TRI_Model * triModel , struct TRI_Model * originalModel , int selectMesh)
{
    int flags = aiProcess_Triangulate;
		flags |= aiProcess_JoinIdenticalVertices;
		flags |= aiProcess_GenSmoothNormals;
		flags |= aiProcess_FlipUVs; //So that I don't need to manually flip UV
		flags |= aiProcess_GenUVCoords;
		flags |= aiProcess_TransformUVCoords;
		flags |= aiProcess_RemoveComponent;

		g_scene = (struct aiScene*) aiImportFile( filename, flags);
		if (g_scene)
        {
            m_GlobalInverseTransform = g_scene->mRootNode->mTransformation;
            m_GlobalInverseTransform.Inverse();

            prepareScene(g_scene,triModel,originalModel,1,selectMesh);

            aiReleaseImport(g_scene);
            return 1;
		} else
		{
			fprintf(stderr, "Assimp Cannot import scene: '%s'\n", filename);
			fprintf(stderr, " error '%s'\n", aiGetErrorString());

		}

  return 0;
}
