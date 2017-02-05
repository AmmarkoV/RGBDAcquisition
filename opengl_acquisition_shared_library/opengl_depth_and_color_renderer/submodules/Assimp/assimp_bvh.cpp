#include "assimp_bvh.h"
#include "assimp_loader.h"

#include <stdio.h>
#include <stdlib.h>


#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

int doBVHConversion(char * sourceBVH)
{
  fprintf(stderr,"BVH converter activated..!\n");
  int flags=0;
  struct aiScene *bvhFile =  (struct aiScene*) aiImportFile( sourceBVH, flags);
		if (bvhFile)
        {
            //m_GlobalInverseTransform = bvhFile->mRootNode->mTransformation;
            //m_GlobalInverseTransform.Inverse();

           // prepareScene(bvhFile,triModel,originalModel,1,selectMesh);

  unsigned int numberOfMeshes=bvhFile->mNumMeshes;
  unsigned int selectedMesh=0;
   for (selectedMesh=0; selectedMesh<numberOfMeshes; selectedMesh++)
     {
  struct aiMesh * mesh = bvhFile->mMeshes[selectedMesh];
      fprintf(stderr,"Mesh #%u (%s)   \n",selectedMesh , mesh->mName.data);
      fprintf(stderr,"  %u vertices \n",mesh->mNumVertices);
      fprintf(stderr,"  %u normals \n",mesh->mNumVertices);
      fprintf(stderr,"  %d faces \n",mesh->mNumFaces);
      fprintf(stderr,"  %d or %d bones\n",mesh->mNumBones,countNumberOfNodes(bvhFile,mesh));
     }

            aiReleaseImport(bvhFile);
            return 1;
		} else
		{
			fprintf(stderr, "Assimp Cannot import bvh file: '%s'\n", sourceBVH);
			fprintf(stderr, " error '%s'\n", aiGetErrorString());

		}

  return 0;
}



