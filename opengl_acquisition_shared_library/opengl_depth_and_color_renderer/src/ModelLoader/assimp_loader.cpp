#include "assimp_loader.h"


#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

struct aiScene *g_scene = NULL;


void extract3x3( aiMatrix3x3 *m3,  aiMatrix4x4 *m4)
{
	m3->a1 = m4->a1; m3->a2 = m4->a2; m3->a3 = m4->a3;
	m3->b1 = m4->b1; m3->b2 = m4->b2; m3->b3 = m4->b3;
	m3->c1 = m4->c1; m3->c2 = m4->c2; m3->c3 = m4->c3;
}

// find a node by name in the hierarchy (for anims and bones)
struct aiNode *findnode(struct aiNode *node, char *name)
{
	int i;
	if (!strcmp(name, node->mName.data))
		return node;
	for (i = 0; i < node->mNumChildren; i++) {
		struct aiNode *found = findnode(node->mChildren[i], name);
		if (found)
			return found;
	}
	return NULL;
}

// calculate absolute transform for node to do mesh skinning
void transformnode(  aiMatrix4x4 *result, struct aiNode *node)
{
	if (node->mParent) {
		transformnode(result, node->mParent);
		aiMultiplyMatrix4(result, &node->mTransformation);
	} else {
		*result = node->mTransformation;
	}
}



void transMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * triModel )
{
    aiMatrix4x4 skin4;
    aiMatrix3x3 skin3;
	unsigned int k;

    struct aiMesh * mesh = scene->mMeshes[meshNumber];
    fprintf(stderr,"  %d bones\n",mesh->mNumBones);

	for (k = 0; k < mesh->mNumBones; k++)
    {
		struct aiBone *bone = mesh->mBones[k];
		struct aiNode *node = findnode(scene->mRootNode, bone->mName.data);

		aiString boneName = bone->mName;
		fprintf(stderr,"Bone %u - %s \n" , k , boneName.C_Str() );


		transformnode(&skin4, node);
		aiMultiplyMatrix4(&skin4, &bone->mOffsetMatrix);
		extract3x3(&skin3, &skin4);

        unsigned int i=0;
		for (i = 0; i < bone->mNumWeights; i++)
        {
			int v = bone->mWeights[i].mVertexId;
			float w = bone->mWeights[i].mWeight;

			aiVector3D position = mesh->mVertices[v];
			aiTransformVecByMatrix4(&position, &skin4);
			triModel->vertices[v*3+0] += position.x * w;
			triModel->vertices[v*3+1] += position.y * w;
			triModel->vertices[v*3+2] += position.z * w;

			aiVector3D normal = mesh->mNormals[v];
			aiTransformVecByMatrix3(&normal, &skin3);
			triModel->normal[v*3+0] += normal.x * w;
			triModel->normal[v*3+1] += normal.y * w;
			triModel->normal[v*3+2] += normal.z * w;
		}

	}

}







void prepareFlatMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * triModel )
{
    struct aiMesh * mesh = scene->mMeshes[meshNumber];

    fprintf(stderr,"Preparing mesh %u   \n",meshNumber);
    fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
    fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
    fprintf(stderr,"  %d faces \n",mesh->mNumFaces);

	unsigned int k;

    fprintf(stderr,"  %d bones\n",mesh->mNumBones);
	for (k = 0; k < mesh->mNumBones; k++)
    {
		aiString boneName = mesh->mBones[k]->mName;
		fprintf(stderr,"   -bone %u - %s \n" , k , boneName.C_Str() );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[0], mesh->mBones[k]->mOffsetMatrix[1], mesh->mBones[k]->mOffsetMatrix[2], mesh->mBones[k]->mOffsetMatrix[3] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[4], mesh->mBones[k]->mOffsetMatrix[5], mesh->mBones[k]->mOffsetMatrix[6], mesh->mBones[k]->mOffsetMatrix[7] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[8], mesh->mBones[k]->mOffsetMatrix[9], mesh->mBones[k]->mOffsetMatrix[10],mesh->mBones[k]->mOffsetMatrix[11] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[12],mesh->mBones[k]->mOffsetMatrix[13], mesh->mBones[k]->mOffsetMatrix[14], mesh->mBones[k]->mOffsetMatrix[15] );
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


    triModel->header.numberOfVertices      = mesh->mNumFaces*3;
    triModel->header.numberOfNormals       = mesh->mNumFaces*3;
    triModel->header.numberOfTextureCoords = mesh->mNumFaces*3;
    triModel->header.numberOfColors        = mesh->mNumFaces*3;
    triModel->header.numberOfIndices       = 0;


	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices * 3 * 3  * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals   * 3 * 3 * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords    * 2 * 3 * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors * 3 * 3  * sizeof(float));
    triModel->indices        = 0;

    unsigned int i=0;

    unsigned int o=0,n=0,t=0,c=0;
	for (i = 0; i < mesh->mNumFaces; i++)
    {
		struct aiFace *face = mesh->mFaces + i;
		unsigned int faceTriA = face->mIndices[0];
		unsigned int faceTriB = face->mIndices[1];
		unsigned int faceTriC = face->mIndices[2];

		//fprintf(stderr,"%u / %u \n" , o , triModel->header.numberOfVertices * 3 );

	    triModel->vertices[o++] = mesh->mVertices[faceTriA].x;
	    triModel->vertices[o++] = mesh->mVertices[faceTriA].y;
	    triModel->vertices[o++] = mesh->mVertices[faceTriA].z;

	    triModel->vertices[o++] = mesh->mVertices[faceTriB].x;
	    triModel->vertices[o++] = mesh->mVertices[faceTriB].y;
	    triModel->vertices[o++] = mesh->mVertices[faceTriB].z;

	    triModel->vertices[o++] = mesh->mVertices[faceTriC].x;
	    triModel->vertices[o++] = mesh->mVertices[faceTriC].y;
	    triModel->vertices[o++] = mesh->mVertices[faceTriC].z;


      if (mesh->mNormals)
        {
			triModel->normal[n++] = mesh->mNormals[faceTriA].x;
			triModel->normal[n++] = mesh->mNormals[faceTriA].y;
			triModel->normal[n++] = mesh->mNormals[faceTriA].z;

			triModel->normal[n++] = mesh->mNormals[faceTriB].x;
			triModel->normal[n++] = mesh->mNormals[faceTriB].y;
			triModel->normal[n++] = mesh->mNormals[faceTriB].z;

			triModel->normal[n++] = mesh->mNormals[faceTriC].x;
			triModel->normal[n++] = mesh->mNormals[faceTriC].y;
			triModel->normal[n++] = mesh->mNormals[faceTriC].z;
		}


      if (mesh->mTextureCoords[0])
        {
			triModel->textureCoords[t++] = mesh->mTextureCoords[0][faceTriA].x;
			triModel->textureCoords[t++] = 1 - mesh->mTextureCoords[0][faceTriA].y;

			triModel->textureCoords[t++] = mesh->mTextureCoords[0][faceTriB].x;
			triModel->textureCoords[t++] = 1 - mesh->mTextureCoords[0][faceTriB].y;

			triModel->textureCoords[t++] = mesh->mTextureCoords[0][faceTriC].x;
			triModel->textureCoords[t++] = 1 - mesh->mTextureCoords[0][faceTriC].y;
		}


        unsigned int colourSet = 0;
        //for (colourSet = 0; colourSet< AI_MAX_NUMBER_OF_COLOR_SETS; colourSet++)
        {
          if(mesh->HasVertexColors(colourSet))
         {
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriA].r;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriA].g;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriA].b;

          triModel->colors[c++] = mesh->mColors[colourSet][faceTriB].r;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriB].g;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriB].b;

          triModel->colors[c++] = mesh->mColors[colourSet][faceTriC].r;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriC].g;
          triModel->colors[c++] = mesh->mColors[colourSet][faceTriC].b;
        }
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

	unsigned int k;
	for (k = 0; k < mesh->mNumBones; k++)
    {
		aiString boneName = mesh->mBones[k]->mName;
		fprintf(stderr,"   -bone %u - %s \n" , k , boneName.C_Str() );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[0], mesh->mBones[k]->mOffsetMatrix[1], mesh->mBones[k]->mOffsetMatrix[2], mesh->mBones[k]->mOffsetMatrix[3] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[4], mesh->mBones[k]->mOffsetMatrix[5], mesh->mBones[k]->mOffsetMatrix[6], mesh->mBones[k]->mOffsetMatrix[7] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[8], mesh->mBones[k]->mOffsetMatrix[9], mesh->mBones[k]->mOffsetMatrix[10],mesh->mBones[k]->mOffsetMatrix[11] );
		fprintf(stderr,"   | %0.2f  | %0.2f  | %0.2f  | %0.2f  | \n" , mesh->mBones[k]->mOffsetMatrix[12],mesh->mBones[k]->mOffsetMatrix[13], mesh->mBones[k]->mOffsetMatrix[14], mesh->mBones[k]->mOffsetMatrix[15] );
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
    triModel->header.numberOfTextureCoords = mesh->mNumVertices*3;
    triModel->header.numberOfColors        = mesh->mNumVertices*3;
    triModel->header.numberOfIndices       = mesh->mNumFaces*3;

   //fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);

	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices * 3 *3  * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals  * 3 *3* sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords    * 2  * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors * 3  *3 * sizeof(float));
    triModel->indices        = (unsigned int*) malloc( triModel->header.numberOfIndices * 3 * sizeof(unsigned int));

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


#define USE_FLAT_CODE 0
void prepareScene(struct aiScene *scene , struct TRI_Model * triModel )
{
    fprintf(stderr,"Preparing scene with %u meshes\n",scene->mNumMeshes);

     #if USE_FLAT_CODE
      prepareFlatMesh(scene, 0,  triModel );
     #else
       struct TRI_Model indexedModel={0};
      prepareMesh(scene, 0,  &indexedModel );
      fillFlatModelTriFromIndexedModelTri(triModel , &indexedModel);
      deallocModelTri(&indexedModel);
    #endif // USE_OK_CODE

    return ;

	unsigned int i=0;
	for (i = 0; i < scene->mNumMeshes; i++)
    {
		prepareMesh(scene, i,  triModel );
		//transformmesh(scene, meshlist + i);
	}

}



int testAssimp(const char * filename  , struct TRI_Model * triModel)
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

            prepareScene(g_scene,triModel);

            return 1;
		} else {
			fprintf(stderr, "cannot import scene: '%s'\n", filename);
		}

  return 0;
}
