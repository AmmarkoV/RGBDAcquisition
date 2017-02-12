#include "model_converter.h"

#include <stdlib.h>


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
