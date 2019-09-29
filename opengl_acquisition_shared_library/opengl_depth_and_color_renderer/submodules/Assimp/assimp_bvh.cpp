#include "assimp_bvh.h"
#include "assimp_loader.h"

#include <stdio.h>
#include <stdlib.h>

//Needs : sudo apt-get install libassimp-dev
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define LIMIT_AT 1000

int doBVHConversion(char * sourceBVH)
{
  fprintf(stderr,"BVH converter activated..!\n");
  int flags=0;
  struct aiScene *bvhFile =  (struct aiScene*) aiImportFile( sourceBVH, flags);
  if (bvhFile)
  {
  fprintf(stderr,"  %u animations in file \n",bvhFile->mNumAnimations);
  fprintf(stderr,"  %u cameras in file \n",bvhFile->mNumCameras);
  fprintf(stderr,"  %u lights in file \n",bvhFile->mNumLights);
  fprintf(stderr,"  %u materials in file \n",bvhFile->mNumMaterials);
  fprintf(stderr,"  %u meshes in file \n",bvhFile->mNumMeshes);
  fprintf(stderr,"  %u textures in file \n",bvhFile->mNumTextures);

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


   unsigned int selectedAnimation=0;
   for (selectedAnimation=0; selectedAnimation<bvhFile->mNumAnimations; selectedAnimation++)
     {
      struct aiAnimation * anim = bvhFile->mAnimations[selectedAnimation];
      fprintf(stderr,"      Animation #%u  (%s) \n",selectedAnimation , anim->mName.data );
      fprintf(stderr,"           %0.2f duration \n",anim->mDuration);
      fprintf(stderr,"           %0.2f ticks per second \n",anim->mTicksPerSecond);
      fprintf(stderr,"           %u channels \n",anim->mNumChannels);
      fprintf(stderr,"           %u Mesh Channels \n",anim->mNumMeshChannels);


       unsigned int selectedNodeOfAnimationChannel=0;


       unsigned int posesToGenerate=0;
       for (selectedNodeOfAnimationChannel=0; selectedNodeOfAnimationChannel<anim->mNumChannels; selectedNodeOfAnimationChannel++)
        {
          struct aiNodeAnim * animNode = anim->mChannels[selectedNodeOfAnimationChannel];

          if (posesToGenerate<animNode->mNumPositionKeys) { posesToGenerate = animNode->mNumPositionKeys; }
          if (posesToGenerate<animNode->mNumRotationKeys) { posesToGenerate = animNode->mNumRotationKeys; }
          if (posesToGenerate<animNode->mNumScalingKeys)  { posesToGenerate = animNode->mNumScalingKeys; }
        }


       for (selectedNodeOfAnimationChannel=0; selectedNodeOfAnimationChannel<posesToGenerate; selectedNodeOfAnimationChannel++)
        {
           fprintf(stdout,"MOVE(human,%u,-19.231,-54.976,2299.735,0.707107,0.707107,0.000000,0.0)\n",selectedNodeOfAnimationChannel);
            if(selectedNodeOfAnimationChannel>LIMIT_AT) { break; }
        }


       for (selectedNodeOfAnimationChannel=0; selectedNodeOfAnimationChannel<anim->mNumChannels; selectedNodeOfAnimationChannel++)
        {
          struct aiNodeAnim * animNode = anim->mChannels[selectedNodeOfAnimationChannel];
          fprintf(stderr,"               Animation Node #%u  (%s) \n",selectedNodeOfAnimationChannel , animNode->mNodeName.data );

          fprintf(stderr,"                     %u positions\n",animNode->mNumPositionKeys);
          fprintf(stderr,"                     %u rotations\n",animNode->mNumRotationKeys);
          fprintf(stderr,"                     %u scalings\n",animNode->mNumScalingKeys);


          unsigned int frameNum=0;
          for (frameNum=0; frameNum<animNode->mNumRotationKeys; frameNum++)
          {
            aiQuatKey * q = &animNode->mRotationKeys[frameNum];
            fprintf(stdout,"POSEQ(human,%u,%s,%0.2f,%0.2f,%0.2f,%0.2f)\n",frameNum,animNode->mNodeName.data,q->mValue.x,q->mValue.y,q->mValue.z,q->mValue.w );
            if(frameNum>LIMIT_AT) { break; }

          }
          /*
aiVectorKey * 	mPositionKeys
 	The position keys of this animation channel.
aiAnimBehaviour 	mPostState
 	Defines how the animation behaves after the last key was processed.
aiAnimBehaviour 	mPreState
 	Defines how the animation behaves before the first key is encountered.
 * 	mRotationKeys
 	The rotation keys of this animation channel.
aiVectorKey * 	mScalingKeys
 	The scaling keys of this animation channel.
*/


        }

      //aiMeshAnim ** 	mMeshChannels

 	  //Duration of the animation in ticks.
 	  //aiMeshAnim ** 	mMeshChannels

 	  break;
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



