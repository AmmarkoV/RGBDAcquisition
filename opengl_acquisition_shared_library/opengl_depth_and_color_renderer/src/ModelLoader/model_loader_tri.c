/** @file model_loader_tri.c
 *  @brief  TRIModels loader/writer and basic functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include "model_loader_tri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#if INCLUDE_OPENGL_CODE
 #include <GL/gl.h>
 #include <GL/glx.h>    /* this includes the necessary X headers */
#endif // INCLUDE_OPENGL_CODE



void print4x4DMatrixTRI(const char * str , double * matrix4x4)
{
  fprintf( stderr, "  %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "  %f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f ",matrix4x4[2]);  fprintf( stderr, "%f\n",matrix4x4[3]);
  fprintf( stderr, "  %f ",matrix4x4[4]);  fprintf( stderr, "%f ",matrix4x4[5]);  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f\n",matrix4x4[7]);
  fprintf( stderr, "  %f ",matrix4x4[8]);  fprintf( stderr, "%f ",matrix4x4[9]);  fprintf( stderr, "%f ",matrix4x4[10]); fprintf( stderr, "%f\n",matrix4x4[11]);
  fprintf( stderr, "  %f ",matrix4x4[12]); fprintf( stderr, "%f ",matrix4x4[13]); fprintf( stderr, "%f ",matrix4x4[14]); fprintf( stderr, "%f\n",matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");

}

void printTRIModel(struct TRI_Model * triModel)
{
 fprintf(stderr,"Number Of Vertices  %u \n",triModel->header.numberOfVertices);
 fprintf(stderr,"Number Of Normals  %u \n",triModel->header.numberOfNormals);
 fprintf(stderr,"Number Of TextureCoords  %u \n",triModel->header.numberOfTextureCoords);
 fprintf(stderr,"Number Of Colors  %u \n",triModel->header.numberOfColors);
 fprintf(stderr,"Number Of Indices  %u \n",triModel->header.numberOfIndices);
 fprintf(stderr,"Number Of Bones  %u \n",triModel->header.numberOfBones);
}

void printTRIBoneStructure(struct TRI_Model * triModel, int alsoPrintMatrices)
{
 unsigned int k=0, i=0 , parent , child ;
   for (i=0; i<triModel->header.numberOfBones; i++)
   {
     fprintf(stderr,"Bone %u : %s \n",i,triModel->bones[i].boneName);
     if (triModel->bones[i].info!=0)
     {
      fprintf(stderr," Weights Number %u \n",triModel->bones[i].info->boneWeightsNumber);
      parent = triModel->bones[i].info->boneParent;
      fprintf(stderr," Parent : %u %s \n",parent,triModel->bones[parent].boneName);

      if (triModel->bones[i].boneChild!=0)
      {
      fprintf(stderr," Children (%u) : " , triModel->bones[i].info->numberOfBoneChildren);
       for (k=0; k<triModel->bones[i].info->numberOfBoneChildren; k++)
        {
         child=triModel->bones[i].boneChild[k];
         fprintf(stderr,"%u %s , ",child,triModel->bones[child].boneName);
        }
       fprintf(stderr,"\n");
      }

     if (alsoPrintMatrices)
        {
         print4x4DMatrixTRI("matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose", triModel->bones[i].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose );
         print4x4DMatrixTRI("finalVertexTransformation", triModel->bones[i].info->finalVertexTransformation );
         print4x4DMatrixTRI("localTransformation", triModel->bones[i].info->localTransformation );
        }
     }
   }
}

int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed)
{
    triModel->header.triType = TRI_LOADER_VERSION;
    triModel->header.floatSize = sizeof(float);
    triModel->header.nameSize = indexed->header.nameSize;
    triModel->header.TRIMagic[0] = 'T';
    triModel->header.TRIMagic[1] = 'R';
    triModel->header.TRIMagic[2] = 'I';
    triModel->header.TRIMagic[3] = '3';
    triModel->header.TRIMagic[4] = 'D';
    triModel->header.numberOfVertices      = indexed->header.numberOfIndices*3;
    triModel->header.numberOfNormals       = indexed->header.numberOfIndices*3;
    triModel->header.numberOfTextureCoords = indexed->header.numberOfIndices*2;
    triModel->header.numberOfColors        = indexed->header.numberOfIndices*3;
    triModel->header.numberOfIndices       = 0;
    triModel->header.numberOfBones         = 0; //indexed->header.numberOfBones;

    fprintf(stderr,"\n\nwarning : Flattening a model loses its bone structure for now .. \n");

    triModel->name = (char * ) malloc( (triModel->header.nameSize+1)  * sizeof(char));
    memcpy(triModel->name , indexed->name , triModel->header.nameSize);
    triModel->name[triModel->header.nameSize]=0;


	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices  *3 *3    * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals   *3 *3     * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords *3 *2  * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors    *3  *3    * sizeof(float));
    triModel->indices        = 0;
    triModel->bones          = 0;
    unsigned int i=0;

    unsigned int o=0,n=0,t=0,c=0;
	for (i = 0; i < indexed->header.numberOfIndices/3; i++)
    {
		unsigned int faceTriA = indexed->indices[(i*3)+0];
		unsigned int faceTriB = indexed->indices[(i*3)+1];
		unsigned int faceTriC = indexed->indices[(i*3)+2];


        unsigned int faceTriA_X = (faceTriA*3)+0;
        unsigned int faceTriA_Y = (faceTriA*3)+1;
        unsigned int faceTriA_Z = (faceTriA*3)+2;

        unsigned int faceTriB_X = (faceTriB*3)+0;
        unsigned int faceTriB_Y = (faceTriB*3)+1;
        unsigned int faceTriB_Z = (faceTriB*3)+2;

        unsigned int faceTriC_X = (faceTriC*3)+0;
        unsigned int faceTriC_Y = (faceTriC*3)+1;
        unsigned int faceTriC_Z = (faceTriC*3)+2;

		//fprintf(stderr,"%u / %u \n" , o , triModel->header.numberOfVertices * 3 );

	    triModel->vertices[o++] = indexed->vertices[faceTriA_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriA_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriA_Z];

	    triModel->vertices[o++] = indexed->vertices[faceTriB_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriB_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriB_Z];

	    triModel->vertices[o++] = indexed->vertices[faceTriC_X];
	    triModel->vertices[o++] = indexed->vertices[faceTriC_Y];
	    triModel->vertices[o++] = indexed->vertices[faceTriC_Z];


      if (indexed->normal)
        {
			triModel->normal[n++] = indexed->normal[faceTriA_X];
			triModel->normal[n++] = indexed->normal[faceTriA_Y];
			triModel->normal[n++] = indexed->normal[faceTriA_Z];

			triModel->normal[n++] = indexed->normal[faceTriB_X];
			triModel->normal[n++] = indexed->normal[faceTriB_Y];
			triModel->normal[n++] = indexed->normal[faceTriB_Z];

			triModel->normal[n++] = indexed->normal[faceTriC_X];
			triModel->normal[n++] = indexed->normal[faceTriC_Y];
			triModel->normal[n++] = indexed->normal[faceTriC_Z];
		}


      if ( indexed->textureCoords)
        {
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriA*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriA*2)+1];

			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriB*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriB*2)+1];

			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriC*2)+0];
			triModel->textureCoords[t++] = indexed->textureCoords[(faceTriC*2)+1];
		}


          if ( indexed->colors )
         {
          triModel->colors[c++] = indexed->colors[faceTriA_X];
          triModel->colors[c++] = indexed->colors[faceTriA_Y];
          triModel->colors[c++] = indexed->colors[faceTriA_Z];

          triModel->colors[c++] = indexed->colors[faceTriB_X];
          triModel->colors[c++] = indexed->colors[faceTriB_Y];
          triModel->colors[c++] = indexed->colors[faceTriB_Z];

          triModel->colors[c++] = indexed->colors[faceTriC_X];
          triModel->colors[c++] = indexed->colors[faceTriC_Y];
          triModel->colors[c++] = indexed->colors[faceTriC_Z];
        }


	}
 return 1;
}



struct TRI_Model * allocateModelTri()
{
  struct TRI_Model * newModel = (struct TRI_Model * ) malloc(sizeof(struct TRI_Model));
  if (newModel!=0) // Clear new model if it was allocated..
        { memset(newModel,0,sizeof(struct TRI_Model)); }
  return (struct TRI_Model * ) newModel;
}


void deallocInternalsOfModelTri(struct TRI_Model * triModel)
{
  if (triModel==0) { return ; }

  triModel->header.numberOfVertices = 0;
  if (triModel->vertices!=0) { free(triModel->vertices); }

  triModel->header.nameSize = 0;
  if (triModel->name!=0) { free(triModel->name); }

  triModel->header.numberOfNormals = 0;
  if (triModel->normal!=0) { free(triModel->normal); }

  triModel->header.numberOfColors = 0;
  if (triModel->colors!=0) { free(triModel->colors); }

  triModel->header.numberOfTextureCoords = 0;
  if (triModel->textureCoords!=0) { free(triModel->textureCoords); }

  triModel->header.numberOfIndices = 0;
  if (triModel->indices!=0) { free(triModel->indices); }

   if (
           (triModel->header.numberOfBones>0) &&
           (triModel->bones!=0)
          )
      {
        unsigned int boneNum =0;
        for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
        {

          if (triModel->bones[boneNum].boneChild!=0)
           { free(triModel->bones[boneNum].boneChild); }

          if (triModel->bones[boneNum].info!=0)
            { free(triModel->bones[boneNum].info); }

          if (triModel->bones[boneNum].boneName!=0)
          { free(triModel->bones[boneNum].boneName); }

          if (triModel->bones[boneNum].weightValue!=0)
          { free(triModel->bones[boneNum].weightValue); }

          if (triModel->bones[boneNum].weightIndex!=0)
          { free(triModel->bones[boneNum].weightIndex); }
        }
      }

   if (triModel->bones!=0)         { free(triModel->bones); }

   memset(triModel,0,sizeof(struct TRI_Model));
}

int freeModelTri(struct TRI_Model * triModel)
{
  if (triModel!=0)
  {
   deallocInternalsOfModelTri(triModel);
   free(triModel);
  }
 return 1;
}


void copyModelTriHeader(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN )
{
  //fprintf(stderr,"Cleaning output model..\n");
  memset(triModelOUT,0,sizeof(struct TRI_Model));
  //fprintf(stderr,"Copying header..\n");
  memcpy(&triModelOUT->header , &triModelIN->header , sizeof(struct TRI_Header));

 return;
}


void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN , int copyBoneStructures)
{
  //fprintf(stderr,MAGENTA "copyModelTri ..\n" NORMAL);
  if (triModelOUT==0) { return; }
  if (triModelIN==0)  { return; }
  copyModelTriHeader( triModelOUT ,  triModelIN );


  unsigned int itemSize , count , allocationSize;

  if (triModelIN->header.nameSize!=0)
  {
   itemSize=sizeof(char); count=triModelIN->header.nameSize; allocationSize = itemSize * count;
   if (triModelOUT->name!=0)  { free(triModelOUT->name); triModelOUT->name=0; }
   if ((triModelIN->name!=0) && (allocationSize>0) )  { triModelOUT->name = (char*) malloc(allocationSize+sizeof(char)); }
   memcpy(triModelOUT->name,triModelIN->name,allocationSize);
   triModelOUT->name[count]=0; //Null terminate
  }

  itemSize=sizeof(float); count=triModelIN->header.numberOfVertices; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of vertices ..\n", allocationSize);
  if (triModelOUT->vertices!=0)  { free(triModelOUT->vertices); triModelOUT->vertices=0; }
  if ((triModelIN->vertices!=0) && (allocationSize>0) )  { triModelOUT->vertices = (float*) malloc(allocationSize); }
  memcpy(triModelOUT->vertices,triModelIN->vertices,allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfNormals; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of normals ..\n", allocationSize);
  if (triModelOUT->normal!=0)  { free(triModelOUT->normal); triModelOUT->normal=0; }
  if ((triModelIN->normal!=0) && (allocationSize>0) )  { triModelOUT->normal = (float*) malloc(allocationSize); }
  memcpy(triModelOUT->normal        , triModelIN->normal        , allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfColors; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of colors ..\n", allocationSize);
  if (triModelOUT->colors!=0)  { free(triModelOUT->colors); triModelOUT->colors=0; }
  if ((triModelIN->colors!=0) && (allocationSize>0) )  { triModelOUT->colors=(float*) malloc(allocationSize); }
  memcpy(triModelOUT->colors        , triModelIN->colors        , allocationSize);

  itemSize=sizeof(float); count=triModelIN->header.numberOfTextureCoords; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of textures ..\n", allocationSize);
  if (triModelOUT->textureCoords!=0)  { free(triModelOUT->textureCoords); triModelOUT->textureCoords=0; }
  if ((triModelIN->textureCoords!=0) && (allocationSize>0) )  { triModelOUT->textureCoords=(float*) malloc(allocationSize); }
  memcpy(triModelOUT->textureCoords , triModelIN->textureCoords , allocationSize);

  itemSize=sizeof(unsigned int); count=triModelIN->header.numberOfIndices; allocationSize = itemSize * count;
  //fprintf(stderr,"Copying %u bytes of indices ..\n", allocationSize);
  if (triModelOUT->indices!=0)  { free(triModelOUT->indices); triModelOUT->indices=0; }
  if ((triModelIN->indices!=0) && (allocationSize>0) )  { triModelOUT->indices = (unsigned int*) malloc(allocationSize); }
  memcpy(triModelOUT->indices , triModelIN->indices       , allocationSize);


  if (triModelOUT->bones!=0)  { free(triModelOUT->bones); triModelOUT->bones=0; }


  if ( (copyBoneStructures) && (triModelIN->header.numberOfBones>0) )
  {
    //fprintf(stderr,GREEN "copyModelTri copying bone structures..\n" NORMAL);
     triModelOUT->bones = (struct TRI_Bones *) malloc(sizeof(struct TRI_Bones) * triModelIN->header.numberOfBones);
     memset(triModelOUT->bones, 0 , sizeof(struct TRI_Bones) * triModelIN->header.numberOfBones);

     unsigned int boneNum=0,itemSize,count;
     for (boneNum=0; boneNum<triModelIN->header.numberOfBones; boneNum++)
        {
         //First read dimensions of bone string and the number of weights for the bone..
         if (triModelOUT->bones[boneNum].info!=0) { free(triModelOUT->bones[boneNum].info); }
         triModelOUT->bones[boneNum].info = (struct TRI_Bones_Header*) malloc(sizeof(struct TRI_Bones_Header));
         memcpy( triModelOUT->bones[boneNum].info , triModelIN->bones[boneNum].info , sizeof(struct TRI_Bones_Header) );

         //Allocate enough space for the bone string , read it  , and null terminate it
         itemSize = sizeof(char);         count = triModelIN->bones[boneNum].info->boneNameSize;
         if (triModelOUT->bones[boneNum].boneName!=0) { free(triModelOUT->bones[boneNum].boneName); }
         triModelOUT->bones[boneNum].boneName = ( char * ) malloc ( (itemSize)*(count+1) );
         memcpy( triModelOUT->bones[boneNum].boneName , triModelIN->bones[boneNum].boneName , itemSize*count );
         triModelOUT->bones[boneNum].boneName[triModelIN->bones[boneNum].info->boneNameSize]=0; //Null terminator..

         //Allocate enough space for the weight values , and read them
         itemSize = sizeof(float);        count = triModelIN->bones[boneNum].info->boneWeightsNumber;
         if (triModelOUT->bones[boneNum].weightValue!=0) { free(triModelOUT->bones[boneNum].weightValue); }
         triModelOUT->bones[boneNum].weightValue = ( float * ) malloc ( itemSize * count );
         memcpy( triModelOUT->bones[boneNum].weightValue , triModelIN->bones[boneNum].weightValue , itemSize * count );

         //Allocate enough space for the weight indexes , and read them
         itemSize = sizeof(unsigned int); count = triModelIN->bones[boneNum].info->boneWeightsNumber;
         if (triModelOUT->bones[boneNum].weightIndex!=0) { free(triModelOUT->bones[boneNum].weightIndex); }
         triModelOUT->bones[boneNum].weightIndex = ( unsigned int * ) malloc ( itemSize * count );
         memcpy( triModelOUT->bones[boneNum].weightIndex , triModelIN->bones[boneNum].weightIndex , itemSize * count );

         itemSize = sizeof(unsigned int); count = triModelIN->bones[boneNum].info->numberOfBoneChildren;
         if (triModelOUT->bones[boneNum].boneChild!=0)
            {
              free(triModelOUT->bones[boneNum].boneChild);
              triModelOUT->bones[boneNum].boneChild=0;
            }
         if ( ( triModelIN->bones[boneNum].boneChild !=0 ) && (count>0) )
          {
            triModelOUT->bones[boneNum].boneChild = (unsigned int *) malloc(  itemSize * (count+1) );
            if (triModelOUT->bones[boneNum].boneChild !=0 )
             {
              memcpy( triModelOUT->bones[boneNum].boneChild , triModelIN->bones[boneNum].boneChild ,  itemSize * count );
             } else
             {
              fprintf(stderr,"Cannot allocate enough memory for bone children..\n");
             }
          }
        }
  } else
  {
    fprintf(stderr,RED "copyModelTri NOT copying bone structures..\n" NORMAL);
    if (triModelOUT->bones!=0)  { free(triModelOUT->bones); triModelOUT->bones=0; }
    triModelOUT->header.numberOfBones=0;
  }

 return ;
}



int loadModelTri(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Reading TRI model -> %s \n",filename );
  FILE *fd=0;
  fd = fopen(filename,"rb");
  if (fd!=0)
    {
        size_t n;
        unsigned int itemSize=0 , count=0;

        n = fread(&triModel->header , sizeof(struct TRI_Header), 1 , fd);
        if (n!= sizeof(struct TRI_Header)) { fprintf(stderr,"Incomplete read of TRI Header\n"); }

        if (triModel->header.floatSize!=sizeof(float))        { fprintf(stderr,"Size of float (%u/%lu) is different , cannot load \n",triModel->header.floatSize,sizeof(float)); return 0; }
        if (triModel->header.triType != TRI_LOADER_VERSION )
            {//TRI_LOADER_VERSION changes can lead to bugs let's have a HUGE warning message
             fprintf(stderr,RED " ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n" NORMAL);
              fprintf(stderr,YELLOW"  Incompatible triloader file , cannot load \n");
              if (triModel->header.triType>TRI_LOADER_VERSION )
                {
                  fprintf(stderr,"  You need to upgrade your TRI Loader code to be able to read it \n");
                } else
              if (triModel->header.triType<TRI_LOADER_VERSION )
                {
                  fprintf(stderr,"  This is an old file-version that was dropped! \n ");
                }
              fprintf(stderr,"   " NORMAL);
             fprintf(stderr,RED " ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n");
             fprintf(stderr," ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! \n" NORMAL);

             fclose(fd);
             return 0;
            }


        //Write file name for internal usage..
        triModel->name = ( char * ) malloc ( sizeof(char) * (triModel->header.nameSize+1) );
        n = fread(triModel->name  , sizeof(char) , triModel->header.nameSize , fd);
        if (n!=sizeof(char) *triModel->header.nameSize) { fprintf(stderr,"Incomplete read of header name field\n"); }

        triModel->name[triModel->header.nameSize]=0;
        fprintf(stderr,"Internal name is -> %s \n",triModel->name);


        if (triModel->header.numberOfVertices)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfVertices;
         fprintf(stderr,"Reading %u bytes of vertex\n", itemSize * count );
         triModel->vertices = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->vertices , itemSize , count , fd);
         if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of vertices\n"); }
        } else {  fprintf(stderr,"No vertices specified \n"); }

        if (triModel->header.numberOfNormals)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfNormals;
         fprintf(stderr,"Reading %u bytes of normal\n", itemSize * count );
         triModel->normal = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->normal , itemSize , count , fd);
         if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of normals\n"); }
        } else {  fprintf(stderr,"No normals specified \n"); }


        if (triModel->header.numberOfTextureCoords)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Reading %u bytes of textures\n",itemSize * count);
         triModel->textureCoords = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->textureCoords , itemSize , count , fd);
         if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of textures\n"); }
        }  else {  fprintf(stderr,"No texture coords specified \n"); }

        if (triModel->header.numberOfColors)
        {
         itemSize=sizeof(float); count=triModel->header.numberOfColors;
         fprintf(stderr,"Reading %u bytes of colors\n",itemSize * count);
         triModel->colors = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->colors ,  itemSize , count , fd);
         if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of colors\n"); }
        } else {  fprintf(stderr,"No colors specified \n"); }

        if (triModel->header.numberOfIndices)
        {
         itemSize=sizeof(unsigned int); count=triModel->header.numberOfIndices;
         fprintf(stderr,"Reading %u bytes of indices\n",itemSize * count);
         triModel->indices = ( unsigned int * ) malloc ( itemSize * count );
         n = fread(triModel->indices , itemSize , count , fd);
         if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of indices\n"); }

        } else {  fprintf(stderr,"No indices specified \n"); }

        if (triModel->header.numberOfBones)
        {
         fprintf(stderr,"Reading %u bones\n",triModel->header.numberOfBones);

         triModel->bones = (struct TRI_Bones *) malloc(sizeof(struct TRI_Bones) * triModel->header.numberOfBones);
         memset(triModel->bones, 0 , sizeof(struct TRI_Bones) * triModel->header.numberOfBones);


         if (triModel->bones)
         {
          unsigned int boneNum=0,itemSize,count;
          for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
          {
          //First read dimensions of bone string and the number of weights for the bone..
          triModel->bones[boneNum].info = (struct TRI_Bones_Header*) malloc(sizeof(struct TRI_Bones_Header));
          memset( triModel->bones[boneNum].info , 0 , sizeof(struct TRI_Bones_Header) );
          n = fread(triModel->bones[boneNum].info , sizeof(struct TRI_Bones_Header), 1 , fd);
          if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of bone header\n"); }

          //Check value
          //fprintf(stderr,"Bone %u \n",boneNum);
          //print4x4DMatrixTRI("triModel->bones[boneNum].info->finalVertexTransformation", triModel->bones[boneNum].info->finalVertexTransformation);

          //Allocate enough space for the bone string , read it  , and null terminate it
          itemSize = sizeof(char);         count = triModel->bones[boneNum].info->boneNameSize;
          triModel->bones[boneNum].boneName = ( char * ) malloc ( (itemSize+2)*count );
          memset( triModel->bones[boneNum].boneName , 0 , (itemSize+2)*count );
          n = fread(triModel->bones[boneNum].boneName , itemSize , count , fd);
          if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of bone names\n"); }


          //Allocate enough space for the weight values , and read them
          itemSize = sizeof(float);        count = triModel->bones[boneNum].info->boneWeightsNumber;
          triModel->bones[boneNum].weightValue = ( float * ) malloc ( itemSize * count );
          memset( triModel->bones[boneNum].weightValue , 0 , itemSize * count );
          n = fread(triModel->bones[boneNum].weightValue , itemSize , count , fd);
          if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of bone weight values\n"); }

          //Allocate enough space for the weight indexes , and read them
          itemSize = sizeof(unsigned int); count = triModel->bones[boneNum].info->boneWeightsNumber;
          triModel->bones[boneNum].weightIndex = ( unsigned int * ) malloc ( itemSize * count );
          memset( triModel->bones[boneNum].weightIndex , 0 , itemSize * count );
          n = fread(triModel->bones[boneNum].weightIndex , itemSize , count , fd);
          if (n!=itemSize*count) { fprintf(stderr,"Incomplete read of bone weight indices\n"); }


          if (triModel->bones[boneNum].info->numberOfBoneChildren == 0 )
          {
           triModel->bones[boneNum].boneChild  = 0;
          } else
          {
           itemSize = sizeof(unsigned int); count = triModel->bones[boneNum].info->numberOfBoneChildren;
           triModel->bones[boneNum].boneChild  = ( unsigned int * ) malloc ( itemSize * count );
           memset( triModel->bones[boneNum].boneChild , 0 , itemSize * count );
           n = fread( triModel->bones[boneNum].boneChild  , itemSize , count , fd);
          }

          }
         }
        } else {  fprintf(stderr,"No bones specified \n"); }


        //printTRIBoneStructure(triModel, 1);
        printTRIModel(triModel);

        fclose(fd);

        return 1;
    }
  return 0;
}


int findTRIBoneWithName(struct TRI_Model * triModel ,const char * searchName , unsigned int * boneIDResult)
{
if ( (triModel->header.numberOfBones) && (triModel->bones!=0) )
  {
   unsigned int boneNum=0;
   for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
     {
        if ( strcmp(searchName,triModel->bones[boneNum].boneName)==0)
            {
              //fprintf(stderr,GREEN "Found bone %s ( %u ) \n" NORMAL , searchName , boneNum);
              *boneIDResult=boneNum;
              return 1;
            }
     }
  }
  fprintf(stderr,RED "Could not find bone %s ( %u bones total ) \n" NORMAL , searchName , triModel->header.numberOfBones);
 return 0;
}



int saveModelTri(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Writing TRI model -> %s \n",filename );
  unsigned int itemSize=0 , count=0;

  FILE * fd = fopen(filename,"wb");
  if (fd!=0)
    {
        triModel->header.nameSize=strlen(filename);
        triModel->header.triType = TRI_LOADER_VERSION;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        triModel->header.TRIMagic[0] = 'T';
        triModel->header.TRIMagic[1] = 'R';
        triModel->header.TRIMagic[2] = 'I';
        triModel->header.TRIMagic[3] = '3';
        triModel->header.TRIMagic[4] = 'D';

        fwrite (&triModel->header , sizeof(struct TRI_Header), 1 , fd);


        //Write file name for internal usage..
        fwrite (filename, sizeof(char), triModel->header.nameSize , fd);

        if ( (triModel->header.numberOfVertices) && (triModel->vertices!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfVertices;
         fprintf(stderr,"Writing %u bytes of vertices\n",itemSize*count);
         fwrite (triModel->vertices , itemSize , count , fd);
        }

        if ( (triModel->header.numberOfNormals) && (triModel->normal!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfNormals;
         fprintf(stderr,"Writing %u bytes of normal\n",itemSize*count);
         fwrite (triModel->normal , itemSize , count  , fd);
        }

        if ( (triModel->header.numberOfTextureCoords) && (triModel->textureCoords!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Writing %u bytes of textureCoords\n",itemSize*count);
         fwrite (triModel->textureCoords,itemSize , count,fd);
        }

        if ( (triModel->header.numberOfColors) && (triModel->colors!=0) )
        {
         itemSize=sizeof(float); count=triModel->header.numberOfColors;
         fprintf(stderr,"Writing %u bytes of colors\n",itemSize*count);
         fwrite (triModel->colors , itemSize , count, fd);
        }

        if ( (triModel->header.numberOfIndices) && (triModel->indices!=0) )
        {
         itemSize=sizeof(unsigned int); count=triModel->header.numberOfIndices;
         fprintf(stderr,"Writing %u bytes of indices\n",itemSize*count);
         fwrite (triModel->indices ,itemSize , count, fd);
        }


        if ( (triModel->header.numberOfBones) && (triModel->bones!=0) )
        {
         struct TRI_Bones_Header emptyBonesHeader={0};
         unsigned int boneNum=0;
         fprintf(stderr,"Writing %u bones\n",triModel->header.numberOfBones);
         for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
         {
          //fprintf(stderr,"%u\n",boneNum);

          if (triModel->bones[boneNum].info!=0)
           { fwrite ( triModel->bones[boneNum].info        , sizeof(struct TRI_Bones_Header) , 1 , fd); } else
           { fwrite ( &emptyBonesHeader                    , sizeof(struct TRI_Bones_Header) , 1 , fd); }

          fwrite ( triModel->bones[boneNum].boneName    , sizeof(char)                    , triModel->bones[boneNum].info->boneNameSize      , fd);

          fwrite ( triModel->bones[boneNum].weightValue , sizeof(float)                   , triModel->bones[boneNum].info->boneWeightsNumber , fd);

          fwrite ( triModel->bones[boneNum].weightIndex , sizeof(unsigned int)            , triModel->bones[boneNum].info->boneWeightsNumber , fd);

          if ( (triModel->bones[boneNum].info->numberOfBoneChildren>0) && (triModel->bones[boneNum].boneChild) )
           { fwrite ( triModel->bones[boneNum].boneChild , sizeof(unsigned int)     , triModel->bones[boneNum].info->numberOfBoneChildren , fd); }

         }
        }


        fflush(fd);
        fclose(fd);
        return 1;
    }
  return 0;
}



//#define INCLUDE_OPENGL_CODE 1
void doTriDrawCalllist(struct TRI_Model * tri )
{
 #if INCLUDE_OPENGL_CODE
  unsigned int i=0,z=0;


  glBegin(GL_TRIANGLES);
    if (tri->header.numberOfIndices > 0 )
    {
     //fprintf(stderr,MAGENTA "drawing indexed TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
     unsigned int faceTriA,faceTriB,faceTriC,faceTriA_X,faceTriA_Y,faceTriA_Z,faceTriB_X,faceTriB_Y,faceTriB_Z,faceTriC_X,faceTriC_Y,faceTriC_Z;

     for (i = 0; i < tri->header.numberOfIndices/3; i++)
     {
      faceTriA = tri->indices[(i*3)+0];      faceTriB = tri->indices[(i*3)+1];      faceTriC = tri->indices[(i*3)+2];
      faceTriA_X = (faceTriA*3)+0;           faceTriA_Y = (faceTriA*3)+1;           faceTriA_Z = (faceTriA*3)+2;
      faceTriB_X = (faceTriB*3)+0;           faceTriB_Y = (faceTriB*3)+1;           faceTriB_Z = (faceTriB*3)+2;
      faceTriC_X = (faceTriC*3)+0;           faceTriC_Y = (faceTriC*3)+1;           faceTriC_Z = (faceTriC*3)+2;

      if (tri->normal)   { glNormal3f(tri->normal[faceTriA_X],tri->normal[faceTriA_Y],tri->normal[faceTriA_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriA_X],tri->colors[faceTriA_Y],tri->colors[faceTriA_Z]);  }
      glVertex3f(tri->vertices[faceTriA_X],tri->vertices[faceTriA_Y],tri->vertices[faceTriA_Z]);

      if (tri->normal)   { glNormal3f(tri->normal[faceTriB_X],tri->normal[faceTriB_Y],tri->normal[faceTriB_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriB_X],tri->colors[faceTriB_Y],tri->colors[faceTriB_Z]);  }
      glVertex3f(tri->vertices[faceTriB_X],tri->vertices[faceTriB_Y],tri->vertices[faceTriB_Z]);

      if (tri->normal)   { glNormal3f(tri->normal[faceTriC_X],tri->normal[faceTriC_Y],tri->normal[faceTriC_Z]); }
      if ( tri->colors ) { glColor3f(tri->colors[faceTriC_X],tri->colors[faceTriC_Y],tri->colors[faceTriC_Z]);  }
      glVertex3f(tri->vertices[faceTriC_X],tri->vertices[faceTriC_Y],tri->vertices[faceTriC_Z]);
	 }
    } else
    {
      //fprintf(stderr,BLUE "drawing flat TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
      for (i=0; i<tri->header.numberOfVertices/3; i++)
        {
         z=(i*3)*3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);

         z+=3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);
         z+=3;
         if (tri->normal) { glNormal3f(tri->normal[z+0],tri->normal[z+1],tri->normal[z+2]); }
         if (tri->colors) { glColor3f(tri->colors[z+0],tri->colors[z+1],tri->colors[z+2]);  }
                            glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);
        }
    }
  glEnd();
 #else
  fprintf(stderr,"OpenGL code not compiled in this model loader TRI code.. \n");
 #endif // INCLUDE_OPENGL_CODE
}



