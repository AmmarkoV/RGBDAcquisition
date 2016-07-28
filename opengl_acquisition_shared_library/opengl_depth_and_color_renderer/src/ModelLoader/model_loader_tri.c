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


	triModel->vertices       = (float*) malloc( triModel->header.numberOfVertices  *3 *3    * sizeof(float));
	triModel->normal         = (float*) malloc( triModel->header.numberOfNormals   *3 *3     * sizeof(float));
	triModel->textureCoords  = (float*) malloc( triModel->header.numberOfTextureCoords *3 *2  * sizeof(float));
    triModel->colors         = (float*) malloc( triModel->header.numberOfColors    *3  *3    * sizeof(float));
    triModel->indices        = 0;

    unsigned int i=0;

    unsigned int o=0,n=0,t=0,c=0;
	for (i = 0; i < indexed->header.numberOfIndices/3; i++)
    {
		unsigned int faceTriA = indexed->indices[(i*3)+0];
		unsigned int faceTriB = indexed->indices[(i*3)+1];
		unsigned int faceTriC = indexed->indices[(i*3)+02];


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
  return (struct TRI_Model * ) newModel;
}

int freeModelTri(struct TRI_Model * triModel)
{
  if (triModel!=0)
  {
      if (triModel->vertices!=0)      { free(triModel->vertices); }
      if (triModel->normal!=0)        { free(triModel->normal); }
      if (triModel->textureCoords!=0) { free(triModel->textureCoords); }
      if (triModel->colors!=0)        { free(triModel->colors); }
      if (triModel->indices!=0)       { free(triModel->indices); }
   free(triModel);
  }

 return 1;
}




int loadModelTri(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Reading TRI model -> %s \n",filename );
  FILE *fd=0;
  fd = fopen(filename,"rb");
  if (fd!=0)
    {
        size_t n;

        n = fread(&triModel->header , sizeof(struct TRI_Header), 1 , fd);
        if (triModel->header.floatSize!=sizeof(float)) { fprintf(stderr,"Size of float (%u/%u) is different , cannot load \n",triModel->header.floatSize,sizeof(float)); return 0; }
        if (triModel->header.triType != TRI_LOADER_VERSION ) { fprintf(stderr,"Incompatible triloader file , cannot load \n",triModel->header.floatSize,sizeof(float)); return 0; }


        if (triModel->header.numberOfVertices)
        {
         fprintf(stderr,"Reading %u bytes of vertex\n",sizeof(float) * 3*3 * triModel->header.numberOfVertices);
         triModel->vertices = ( float * ) malloc ( sizeof(float) * 3*3 * triModel->header.numberOfVertices );
         n = fread(triModel->vertices , sizeof(float), 3 * triModel->header.numberOfVertices , fd);
        } else {  fprintf(stderr,"No vertices specified \n"); }

        if (triModel->header.numberOfNormals)
        {
         fprintf(stderr,"Reading %u bytes of normal\n",sizeof(float) * 3*3 * triModel->header.numberOfNormals);
         triModel->normal = ( float * ) malloc ( sizeof(float) * 3*3 * triModel->header.numberOfNormals );
         n = fread(triModel->normal , sizeof(float), 3 * triModel->header.numberOfNormals , fd);
        } else {  fprintf(stderr,"No normals specified \n"); }


        if (triModel->header.numberOfTextureCoords)
        {
        fprintf(stderr,"Reading %u bytes of textures\n",sizeof(float) * 3*2 *triModel->header.numberOfTextureCoords);
        triModel->textureCoords = ( float * ) malloc ( sizeof(float) * 3*2 * triModel->header.numberOfTextureCoords );
        n = fread(triModel->textureCoords , sizeof(float), 2 * triModel->header.numberOfTextureCoords , fd);
        }  else {  fprintf(stderr,"No texture coords specified \n"); }

        if (triModel->header.numberOfColors)
        {
         fprintf(stderr,"Reading %u bytes of colors\n",sizeof(float) * 3*3 *triModel->header.numberOfColors);
         triModel->colors = ( float * ) malloc ( sizeof(float) * 3*3 * triModel->header.numberOfColors );
         n = fread(triModel->colors , sizeof(float), 3 * triModel->header.numberOfColors , fd);
        } else {  fprintf(stderr,"No colors specified \n"); }

        if (triModel->header.numberOfIndices)
        {
         fprintf(stderr,"Reading %u bytes of indices\n",sizeof(unsigned int) * 3*3 *triModel->header.numberOfIndices);
         triModel->indices = ( unsigned int * ) malloc ( sizeof(unsigned int) * 3*3 * triModel->header.numberOfIndices );
         n = fread(triModel->indices , sizeof(unsigned int), 3 * triModel->header.numberOfIndices , fd);
        } else {  fprintf(stderr,"No indices specified \n"); }


        fclose(fd);
        return 1;
    }
  return 0;
}




int saveModelTri(const char * filename , struct TRI_Model * triModel)
{
  fprintf(stderr,"Writing TRI model -> %s \n",filename );
  unsigned int i=0;
  FILE *fd=0;
  fd = fopen(filename,"wb");
  if (fd!=0)
    {
        triModel->header.triType = TRI_LOADER_VERSION;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        fwrite (&triModel->header        , sizeof(struct TRI_Header), 1 , fd);

        if (triModel->header.numberOfVertices)
        {
         fprintf(stderr,"Writing %u bytes of vertex\n", sizeof(float)  * 3  * triModel->header.numberOfVertices);
         fwrite (triModel->vertices ,  3*sizeof(float), triModel->header.numberOfVertices, fd);
        }

        if (triModel->header.numberOfNormals)
        {
        fprintf(stderr,"Writing %u bytes of normal\n",sizeof(float)  * 3 * triModel->header.numberOfNormals);
        fwrite (triModel->normal         ,  3*sizeof(float), triModel->header.numberOfNormals  , fd);
        }

        if (triModel->header.numberOfTextureCoords)
        {
        fprintf(stderr,"Writing %u bytes of texture coords\n", sizeof(float) * 2 * triModel->header.numberOfTextureCoords);
        fwrite (triModel->textureCoords , 2*sizeof(float), triModel->header.numberOfTextureCoords, fd);
        }

        if (triModel->header.numberOfColors)
        {
        fprintf(stderr,"Writing %u bytes of colors\n", sizeof(float)  * 3 * triModel->header.numberOfColors);
        fwrite (triModel->colors , 3*sizeof(float), triModel->header.numberOfColors, fd);
        }

        if (triModel->header.numberOfIndices)
        {
        fprintf(stderr,"Writing %u bytes of indices\n", sizeof(unsigned int)  * 3 * triModel->header.numberOfIndices);
        fwrite (triModel->indices , 3*sizeof(unsigned int ), triModel->header.numberOfIndices, fd);
        }

        fflush(fd);
        fclose(fd);
        return 1;
    }
  return 0;
}

void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN )
{
  unsigned int bufSize;

  memset(triModelOUT,0,sizeof(struct TRI_Model));
  memcpy(&triModelOUT->header , &triModelIN->header , sizeof(struct TRI_Header));


  bufSize = sizeof(float)  * 3 * triModelIN->header.numberOfVertices ;
  if (triModelOUT->vertices!=0)  { free(triModelOUT->vertices); }
  if (triModelIN->vertices!=0)   { triModelOUT->vertices = (float*) malloc(bufSize); }
  memcpy(triModelOUT->vertices,triModelIN->vertices,bufSize);

  bufSize =  sizeof(float)  * 3 * triModelIN->header.numberOfNormals;
  if (triModelOUT->normal!=0)  { free(triModelOUT->normal); }
  if (triModelIN->normal!=0)   { triModelOUT->normal = (float*) malloc(bufSize); }
  memcpy(triModelOUT->normal        , triModelIN->normal        , bufSize);

  bufSize = sizeof(float) * 3 * triModelIN->header.numberOfColors;
  if (triModelOUT->colors!=0)  { free(triModelOUT->colors); }
  if (triModelIN->colors!=0)   { triModelOUT->colors=(float*) malloc(bufSize); }
  memcpy(triModelOUT->colors        , triModelIN->colors        , bufSize);

  bufSize = sizeof(float)  * 2 * triModelIN->header.numberOfTextureCoords;
  if (triModelOUT->textureCoords!=0)  { free(triModelOUT->textureCoords); }
  if (triModelIN->textureCoords!=0)   { triModelOUT->textureCoords=(float*) malloc(bufSize); }
  memcpy(triModelOUT->textureCoords , triModelIN->textureCoords , bufSize);

  bufSize = sizeof(unsigned int)  * 3 * triModelIN->header.numberOfIndices;
  if (triModelOUT->indices!=0)  { free(triModelOUT->indices); }
  if (triModelIN->indices!=0)   { triModelOUT->indices = (unsigned int*) malloc(bufSize); }
  memcpy(triModelOUT->indices       , triModelIN->indices       , bufSize);
}


void deallocModelTri(struct TRI_Model * triModel)
{
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
}


/*
int saveModelTriHeader(const char * filename , struct TRI_Model * triModel)
{

  char headerOut[256];
  snprintf(headerOut,256,"%s.h",filename);

  unsigned int i=0;
  FILE *fd=0;
  fd = fopen(headerOut,"w");
  if (fd!=0)
    {

        fprintf(fd,"const float %sVertices[] = { \n",filename);
        for (i=0; i<triModel->header.numberOfVertices; i++)
        {
          fprintf(
                   fd,"%0.4f , %0.4f , %0.4f ",
                   triModel->vertices[(i*3)+0],
                   triModel->vertices[(i*3)+1],
                   triModel->vertices[(i*3)+2]
                  );

          if (i+1<triModel->header.numberOfVertices) { fprintf(fd,", \n"); }
        }
        fprintf(fd,"}; \n\n");


        fprintf(fd,"const float %sNormals[] = { \n",filename);
        for (i=0; i<triModel->header.numberOfNormals; i++)
        {
           fprintf(
                   fd,"%0.4f , %0.4f , %0.4f ",
                   triModel->normal[(i*3)+0],
                   triModel->normal[(i*3)+1],
                   triModel->normal[(i*3)+2]
                  );
          if (i+1<triModel->header.numberOfNormals) { fprintf(fd,", \n"); }
        }
        fprintf(fd,"}; \n\n");

       fflush(fd);
       fclose(fd);
       return 1;
    }
  return 0;
}
*/




void doTriDrawCalllist(struct TRI_Model * tri )
{
 #if INCLUDE_OPENGL_CODE
  glBegin(GL_TRIANGLES);

 unsigned int i=0,z=0;
      for (i=0; i<tri->header.numberOfVertices/3; i++)
        {
                      glNormal3f(tri->normal[i+0],tri->normal[i+1],tri->normal[i+2]);
          z=(i*3)*3;  glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);

                      glNormal3f(tri->normal[i+0],tri->normal[i+1],tri->normal[i+2]);
          z+=3;       glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);

                      glNormal3f(tri->normal[i+0],tri->normal[i+1],tri->normal[i+2]);
          z+=3;       glVertex3f(tri->vertices[z+0],tri->vertices[z+1],tri->vertices[z+2]);
        }


  glEnd();
 #else
  fprintf(stderr,"OpenGL code not compiled in this model loader TRI code.. \n");
 #endif // INCLUDE_OPENGL_CODE
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
       tri->header.drawType = 0;          // Triangles

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


