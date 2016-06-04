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



/*



    aiMatrix4x4 skin4;
    aiMatrix3x3 skin3;
	unsigned int k;

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
			triModel->triangleVertex[v*3+0] += position.x * w;
			triModel->triangleVertex[v*3+1] += position.y * w;
			triModel->triangleVertex[v*3+2] += position.z * w;

			aiVector3D normal = mesh->mNormals[v];
			aiTransformVecByMatrix3(&normal, &skin3);
			triModel->normal[v*3+0] += normal.x * w;
			triModel->normal[v*3+1] += normal.y * w;
			triModel->normal[v*3+2] += normal.z * w;
		}

	}


*/







void prepareMesh(struct aiScene *scene , int meshNumber , struct TRI_Model * triModel )
{
    struct aiMesh * mesh = scene->mMeshes[meshNumber];

    fprintf(stderr,"Preparing mesh %u   \n",meshNumber);
    fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
    fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
    fprintf(stderr,"  %d faces \n",mesh->mNumFaces);


    aiMatrix4x4 skin4;
    aiMatrix3x3 skin3;
	unsigned int k;

    fprintf(stderr,"  %d bones\n",mesh->mNumBones);
	for (k = 0; k < mesh->mNumBones; k++)
    {
		aiString boneName = mesh->mBones[k]->mName;
		fprintf(stderr,"   -bone %u - %s \n" , k , boneName.C_Str() );
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


    triModel->header.numberOfTriangles = mesh->mNumFaces*3;
    triModel->header.numberOfNormals   = mesh->mNumFaces*3;
    triModel->header.numberOfColors    = mesh->mNumFaces*3;

	triModel->triangleVertex = (float*) malloc( triModel->header.numberOfTriangles * 3 * 3  * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals   * 3 * 3 * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTriangles    * 2 * 3 * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors * 3 * 3  * sizeof(float));

    unsigned int i=0;

    unsigned int o=0,n=0,t=0,c=0;
	for (i = 0; i < mesh->mNumFaces; i++)
    {
		struct aiFace *face = mesh->mFaces + i;
		unsigned int faceTriA = face->mIndices[0];
		unsigned int faceTriB = face->mIndices[1];
		unsigned int faceTriC = face->mIndices[2];

		//fprintf(stderr,"%u / %u \n" , o , triModel->header.numberOfTriangles * 3 );

	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriA].x;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriA].y;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriA].z;

	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriB].x;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriB].y;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriB].z;

	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriC].x;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriC].y;
	    triModel->triangleVertex[o++] = mesh->mVertices[faceTriC].z;


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


void prepareScene(struct aiScene *scene , struct TRI_Model * triModel )
{

    fprintf(stderr,"Preparing scene with %u meshes\n",scene->mNumMeshes);
    prepareMesh(scene, 0,  triModel );
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
		} else {
			fprintf(stderr, "cannot import scene: '%s'\n", filename);
		}














}
