#include "model_loader_tri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if INCLUDE_OPENGL_CODE
 #include <GL/gl.h>
 #include <GL/glx.h>    /* this includes the necessary X headers */
#endif // INCLUDE_OPENGL_CODE




int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed)
{

    triModel->header.numberOfVertices      = indexed->header.numberOfIndices*3;
    triModel->header.numberOfNormals       = indexed->header.numberOfIndices*3;
    triModel->header.numberOfTextureCoords = indexed->header.numberOfIndices*2;
    triModel->header.numberOfColors        = indexed->header.numberOfIndices*3;
    triModel->header.numberOfIndices       = 0;
    triModel->header.numberOfBones         = 0; //indexed->header.numberOfBones;

    fprintf(stderr,"\n\nwarning : Flattening a model loses its bone structure for now .. \n");

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
}



struct TRI_Model * allocateModelTri()
{
  struct TRI_Model * newModel = (struct TRI_Model * ) malloc(sizeof(struct TRI_Model));
  if (newModel!=0) // Clear new model if it was allocated..
        { memset(newModel,0,sizeof(struct TRI_Model)); }
  return (struct TRI_Model * ) newModel;
}


void deallocModelTri(struct TRI_Model * triModel)
{
  if (triModel==0) { return ; }

  triModel->header.numberOfVertices = 0;
  if (triModel->vertices!=0) { free(triModel->vertices); }

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
          free(triModel->bones[boneNum].boneName);
          free(triModel->bones[boneNum].weightValue);
          free(triModel->bones[boneNum].weightIndex);
        }
      }

   if (triModel->bones!=0)         { free(triModel->bones); }
}

int freeModelTri(struct TRI_Model * triModel)
{
  if (triModel!=0)
  {
   deallocModelTri(triModel);
   free(triModel);
  }
 return 1;
}


void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN )
{
  fprintf(stderr,"copyModelTri ignores bone structures..\n");
  triModelOUT->bones=0;
  triModelOUT->header.numberOfBones=0;


  unsigned int itemSize , count;

  memset(triModelOUT,0,sizeof(struct TRI_Model));
  memcpy(&triModelOUT->header , &triModelIN->header , sizeof(struct TRI_Header));

  itemSize=sizeof(float)*3; count=triModelIN->header.numberOfVertices;
  if (triModelOUT->vertices!=0)  { free(triModelOUT->vertices); }
  if (triModelIN->vertices!=0)   { triModelOUT->vertices = (float*) malloc(itemSize*count); }
  memcpy(triModelOUT->vertices,triModelIN->vertices,itemSize*count);

  itemSize=sizeof(float)*3; count=triModelIN->header.numberOfNormals;
  if (triModelOUT->normal!=0)  { free(triModelOUT->normal); }
  if (triModelIN->normal!=0)   { triModelOUT->normal = (float*) malloc(itemSize*count); }
  memcpy(triModelOUT->normal        , triModelIN->normal        , itemSize*count);

  itemSize=sizeof(float)*3; count=triModelIN->header.numberOfColors;
  if (triModelOUT->colors!=0)  { free(triModelOUT->colors); }
  if (triModelIN->colors!=0)   { triModelOUT->colors=(float*) malloc(itemSize*count); }
  memcpy(triModelOUT->colors        , triModelIN->colors        , itemSize*count);

  itemSize=sizeof(float)*2; count=triModelIN->header.numberOfTextureCoords;
  if (triModelOUT->textureCoords!=0)  { free(triModelOUT->textureCoords); }
  if (triModelIN->textureCoords!=0)   { triModelOUT->textureCoords=(float*) malloc(itemSize*count); }
  memcpy(triModelOUT->textureCoords , triModelIN->textureCoords , itemSize*count);

  itemSize=sizeof(unsigned int)*3; count=triModelIN->header.numberOfIndices;
  if (triModelOUT->indices!=0)  { free(triModelOUT->indices); }
  if (triModelIN->indices!=0)   { triModelOUT->indices = (unsigned int*) malloc(itemSize*count); }
  memcpy(triModelOUT->indices       , triModelIN->indices       , itemSize*count);

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
        if (triModel->header.floatSize!=sizeof(float)) { fprintf(stderr,"Size of float (%u/%u) is different , cannot load \n",triModel->header.floatSize,sizeof(float)); return 0; }
        if (triModel->header.triType != TRI_LOADER_VERSION )  { fprintf(stderr,"Incompatible triloader file , cannot load \n",triModel->header.floatSize,sizeof(float)); return 0; }


        if (triModel->header.numberOfVertices)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfVertices;
         fprintf(stderr,"Reading %u bytes of vertex\n", itemSize * count );
         triModel->vertices = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->vertices , itemSize , count , fd);
        } else {  fprintf(stderr,"No vertices specified \n"); }

        if (triModel->header.numberOfNormals)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfNormals;
         fprintf(stderr,"Reading %u bytes of normal\n", itemSize * count );
         triModel->normal = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->normal , itemSize , count , fd);
        } else {  fprintf(stderr,"No normals specified \n"); }


        if (triModel->header.numberOfTextureCoords)
        {
         itemSize=sizeof(float)*2; count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Reading %u bytes of textures\n",itemSize * count);
         triModel->textureCoords = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->textureCoords , itemSize , count , fd);
        }  else {  fprintf(stderr,"No texture coords specified \n"); }

        if (triModel->header.numberOfColors)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfColors;
         fprintf(stderr,"Reading %u bytes of colors\n",itemSize * count);
         triModel->colors = ( float * ) malloc ( itemSize * count );
         n = fread(triModel->colors ,  itemSize , count , fd);
        } else {  fprintf(stderr,"No colors specified \n"); }

        if (triModel->header.numberOfIndices)
        {
         itemSize=sizeof(unsigned int)*3; count=triModel->header.numberOfIndices;
         fprintf(stderr,"Reading %u bytes of indices\n",itemSize * count);
         triModel->indices = ( unsigned int * ) malloc ( itemSize * count );
         n = fread(triModel->indices , itemSize , count , fd);
        } else {  fprintf(stderr,"No indices specified \n"); }

        if (triModel->header.numberOfBones)
        {
         fprintf(stderr,"Reading %u bones\n",triModel->header.numberOfBones);

         triModel->bones = (struct TRI_Bones *) malloc(sizeof(struct TRI_Bones) * triModel->header.numberOfBones);
         if (triModel->bones)
         {
          unsigned int boneNum=0;
          for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
          {
          //First read dimensions of bone string and the number of weights for the bone..
          fprintf(stderr,"Reading header for bone %u \n",boneNum);
          n = fread(&triModel->bones[boneNum].info , sizeof(struct TRI_Bones_Header), 1 , fd);

          //Allocate enough space for the bone string , read it  , and null terminate it

          fprintf(stderr,"Allocating space for name %u \n",triModel->bones[boneNum].info.boneNameSize);
          triModel->bones[boneNum].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneNum].info.boneNameSize+1) );
          n = fread(triModel->bones[boneNum].boneName , sizeof(char), triModel->bones[boneNum].info.boneNameSize , fd);
          triModel->bones[boneNum].boneName[triModel->bones[boneNum].info.boneNameSize]=0;
          fprintf(stderr,"Bone Name is %s \n",triModel->bones[boneNum].boneName);

          //Allocate enough space for the weight values , and read them
          fprintf(stderr,"Allocating space for weights %u \n",triModel->bones[boneNum].info.boneWeightsNumber);
          triModel->bones[boneNum].weightValue = ( char * ) malloc ( sizeof(triModel->bones[boneNum].weightValue) * (triModel->bones[boneNum].info.boneWeightsNumber) );
          n = fread(triModel->bones[boneNum].weightValue , sizeof(triModel->bones[boneNum].weightValue) , triModel->bones[boneNum].info.boneWeightsNumber , fd);

          //Allocate enough space for the weight indexes , and read them
          triModel->bones[boneNum].weightIndex = ( char * ) malloc ( sizeof(triModel->bones[boneNum].weightIndex) * (triModel->bones[boneNum].info.boneWeightsNumber) );
          n = fread(triModel->bones[boneNum].weightIndex , sizeof(triModel->bones[boneNum].weightIndex) , triModel->bones[boneNum].info.boneWeightsNumber , fd);
          }
         }


        } else {  fprintf(stderr,"No bones specified \n"); }


        fclose(fd);
        return 1;
    }
  return 0;
}




int saveModelTri(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Writing TRI model -> %s \n",filename );
  unsigned int i=0 , itemSize=0 , count=0;

  FILE * fd = fopen(filename,"wb");
  if (fd!=0)
    {
        triModel->header.triType = TRI_LOADER_VERSION;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        fwrite (&triModel->header , sizeof(struct TRI_Header), 1 , fd);

        if (triModel->header.numberOfVertices)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfVertices;
         fprintf(stderr,"Writing %u bytes of vertex\n",itemSize*count);
         fwrite (triModel->vertices , itemSize , count , fd);
        }

        if (triModel->header.numberOfNormals)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfNormals;
         fprintf(stderr,"Writing %u bytes of normal\n",itemSize*count);
         fwrite (triModel->normal , itemSize , count  , fd);
        }

        if (triModel->header.numberOfTextureCoords)
        {
         itemSize=sizeof(float)*2; count=triModel->header.numberOfTextureCoords;
         fprintf(stderr,"Writing %u bytes of texture coords\n",itemSize*count);
         fwrite (triModel->textureCoords,itemSize , count,fd);
        }

        if (triModel->header.numberOfColors)
        {
         itemSize=sizeof(float)*3; count=triModel->header.numberOfColors;
         fprintf(stderr,"Writing %u bytes of colors\n",itemSize*count);
         fwrite (triModel->colors , itemSize , count, fd);
        }

        if (triModel->header.numberOfIndices)
        {
         itemSize=sizeof(unsigned int)*3; count=triModel->header.numberOfIndices;
         fprintf(stderr,"Writing %u bytes of indices\n",itemSize*count);
         fwrite (triModel->indices ,itemSize , count, fd);
        }


        if (triModel->header.numberOfBones)
        {
         fprintf(stderr,"Writing %u bones\n",triModel->header.numberOfBones);

         unsigned int boneNum=0;
         for (boneNum=0; boneNum<triModel->header.numberOfBones; boneNum++)
         {
          fwrite (&triModel->bones[boneNum].info , sizeof(struct TRI_Bones_Header) , 1 , fd);
          fwrite (triModel->bones[boneNum].boneName , sizeof(char) , triModel->bones[boneNum].info.boneNameSize , fd);
          fwrite (triModel->bones[boneNum].weightValue , sizeof(triModel->bones[boneNum].weightValue) , triModel->bones[boneNum].info.boneWeightsNumber , fd);
          fwrite (triModel->bones[boneNum].weightIndex , sizeof(triModel->bones[boneNum].weightIndex) , triModel->bones[boneNum].info.boneWeightsNumber , fd);
         }
        }


        fflush(fd);
        fclose(fd);
        return 1;
    }
  return 0;
}






#define HAVE_OBJ_CODE_AVAILIABLE 1

#if HAVE_OBJ_CODE_AVAILIABLE
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj)
{
   if (tri==0) { return 0; }
   if (obj==0) { return 0; }

   unsigned int i=0,j=0,pos=0,posTex=0;

       tri->header.triType=TRI_LOADER_VERSION;
       tri->header.numberOfVertices      = obj->numGroups * obj->numFaces * 3 ;
       tri->header.numberOfNormals       = obj->numGroups * obj->numFaces * 3 ;
       tri->header.numberOfColors        = obj->numGroups * obj->numFaces * 3 ;
       tri->header.numberOfTextureCoords = obj->numGroups * obj->numFaces * 2 ;
       tri->header.numberOfIndices   = 0; // We go full flat when converting an obj image
       tri->indices                  = 0; // We go full flat when converting an obj image
       tri->header.numberOfBones     = 0; //Obj files dont have bones
       tri->bones                    = 0; //Obj files dont have bones
       tri->header.drawType = 0;     // Triangles

       tri->textureCoords = malloc(sizeof(float) *3 *2 * tri->header.numberOfVertices);
       tri->vertices      = malloc(sizeof(float) *3 *3*  tri->header.numberOfVertices);
       tri->normal        = malloc(sizeof(float) *3 *3*  tri->header.numberOfNormals);
       tri->colors        = malloc(sizeof(float) *3 *3*  tri->header.numberOfColors);


       for(i=0; i<obj->numGroups; i++)
	   {
        for(j=0; j<obj->groups[i].numFaces; j++)
			{
			  tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].x;
			  tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].y;
              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].z;

              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].x;
              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].y;
              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].z;

              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].x;
              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].y;
              tri->vertices[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].z;
			}
		}

        pos=0;
        for(i=0; i<obj->numGroups; i++)
		{
         for(j=0; j<obj->groups[i].numFaces; j++)
			{
              if( obj->groups[i].hasNormals)
                  {
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n1;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n2;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n3;

                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n1;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n2;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n3;


                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n1;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n2;
                     tri->normal[pos++] = obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n3;
                  }
				else
				 {
					tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3;

                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3;

                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2;
                    tri->normal[pos++] = obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3;
				}
			}//FOR J
		}

      if (obj->texList!=0)
      {
      posTex=0;
	  for(i=0; i<obj->numGroups; i++)
	   {
        for(j=0; j<obj->groups[i].numFaces; j++)
			{
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[0]].u;
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[0]].v;
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[1]].u;
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[1]].v;
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[2]].u;
              tri->textureCoords[posTex++] =  obj->texList[ obj->faceList[ obj->groups[i].faceList[j]].t[2]].v;
			}
		}
      }



      pos=0;
      if (obj->colorList!=0)
      {
       for(i=0; i<obj->numGroups; i++)
	   {
        for(j=0; j<obj->groups[i].numFaces; j++)
			{
			  tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].r;
			  tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].g;
              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].b;

              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].r;
              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].g;
              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].b;

              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].r;
              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].g;
              tri->colors[pos++] = obj->colorList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].b;
			}
		}
      }

  return 1;
}
#endif // HAVE_OBJ_CODE_AVAILIABLE


//#define INCLUDE_OPENGL_CODE 1
void doTriDrawCalllist(struct TRI_Model * tri )
{
 #if INCLUDE_OPENGL_CODE
  unsigned int i=0,z=0;

  glBegin(GL_TRIANGLES);
    if (tri->header.numberOfIndices > 0 )
    {
     unsigned int faceTriA,faceTriB,faceTriC,faceTriA_X,faceTriA_Y,faceTriA_Z,faceTriB_X,faceTriB_Y,faceTriB_Z,faceTriC_X,faceTriC_Y,faceTriC_Z;

     for (i = 0; i < tri->header.numberOfIndices/3; i++)
     {
      faceTriA = tri->indices[(i*3)+0];      faceTriB = tri->indices[(i*3)+1];      faceTriC = tri->indices[(i*3)+2];
      faceTriA_X = (faceTriA*3)+0;           faceTriA_Y = (faceTriA*3)+1;           faceTriA_Z = (faceTriA*3)+2;
      faceTriB_X = (faceTriB*3)+0;           faceTriB_Y = (faceTriB*3)+1;           faceTriB_Z = (faceTriB*3)+2;
      faceTriC_X = (faceTriC*3)+0;           faceTriC_Y = (faceTriC*3)+1;           faceTriC_Z = (faceTriC*3)+2;

      if (tri->normal)
        { glNormal3f(tri->normal[faceTriA_X],tri->normal[faceTriA_Y],tri->normal[faceTriA_Z]); }
      if ( tri->colors )
        { glColor3f(tri->colors[faceTriA_X],tri->colors[faceTriA_Y],tri->colors[faceTriA_Z]);  }
      glVertex3f(tri->vertices[faceTriA_X],tri->vertices[faceTriA_Y],tri->vertices[faceTriA_Z]);

      if (tri->normal)
        { glNormal3f(tri->normal[faceTriB_X],tri->normal[faceTriB_Y],tri->normal[faceTriB_Z]); }
      if ( tri->colors )
        { glColor3f(tri->colors[faceTriB_X],tri->colors[faceTriB_Y],tri->colors[faceTriB_Z]);  }
      glVertex3f(tri->vertices[faceTriB_X],tri->vertices[faceTriB_Y],tri->vertices[faceTriB_Z]);

      if (tri->normal)
        { glNormal3f(tri->normal[faceTriC_X],tri->normal[faceTriC_Y],tri->normal[faceTriC_Z]); }
      if ( tri->colors )
        { glColor3f(tri->colors[faceTriC_X],tri->colors[faceTriC_Y],tri->colors[faceTriC_Z]);  }
      glVertex3f(tri->vertices[faceTriC_X],tri->vertices[faceTriC_Y],tri->vertices[faceTriC_Z]);
	 }
    } else
    {
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


