#include "model_loader_tri.h"
#include "model_loader_obj.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj)
{
   if (tri==0) { return 0; }
   if (obj==0) { return 0; }

   unsigned int i=0,j=0,pos=0;

       tri->triType=1;
       tri->numberOfTriangles = obj->numGroups * obj->numFaces * 3 ;
       tri->numberOfNormals   = obj->numGroups * obj->numFaces * 3 ;

       tri->triangleVertex = malloc(sizeof(float) * 3 * tri->numberOfTriangles);
       for(i=0; i<obj->numGroups; i++)
	   {
        for(j=0; j<obj->groups[i].numFaces; j++)
			{
			  tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].x;
			  tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].y;
              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].z;

              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].x;
              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].y;
              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].z;

              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].x;
              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].y;
              tri->triangleVertex[pos++] = obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].z;
			}
		}



        pos=0;
        tri->normal = malloc(sizeof(float) * 3 * tri->numberOfNormals);
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
  return 1;
}


int loadModelTri(const char * filename , struct TRI_Model * triModel)
{
  FILE *fd=0;
  unsigned int floatSize;
  fd = fopen(filename,"rb");
  if (fd!=0)
    {
        unsigned int i=0;
        fscanf(fd,"%u\n",&floatSize);
        if (floatSize!=sizeof(float)) { fprintf(stderr,"Size of float is different , cannot load \n"); return 0; }

        fscanf(fd,"%u\n",&triModel->numberOfTriangles );
        triModel->triangleVertex = malloc(sizeof(float) * 3 * triModel->numberOfTriangles);
        for (i=0; i<triModel->numberOfTriangles; i++)
        {
          fscanf(
                   fd,"%f%f%f",
                   &triModel->triangleVertex[(i*3)+0],
                   &triModel->triangleVertex[(i*3)+1],
                   &triModel->triangleVertex[(i*3)+2]
                  );
        }


        fscanf(fd,"%u\n",&triModel->numberOfNormals );
        triModel->normal = malloc(sizeof(float) * 3 * triModel->numberOfNormals);
        for (i=0; i<triModel->numberOfNormals; i++)
        {
          fscanf(
                   fd,"%f%f%f",
                   &triModel->normal[(i*3)+0],
                   &triModel->normal[(i*3)+1],
                   &triModel->normal[(i*3)+2]
                  );

        }

        fclose(fd);
        return 1;
    }
  return 0;
}




int saveModelTri(const char * filename , struct TRI_Model * triModel)
{
  unsigned int i=0;
  FILE *fd=0;
  fd = fopen(filename,"wb");
  if (fd!=0)
    {
        fprintf(fd,"%u\n",sizeof(float));
        fprintf(fd,"%u\n",triModel->numberOfTriangles );
        for (i=0; i<triModel->numberOfTriangles; i++)
        {
          fprintf(
                   fd,"%f%f%f",
                   triModel->triangleVertex[(i*3)+0],
                   triModel->triangleVertex[(i*3)+1],
                   triModel->triangleVertex[(i*3)+2]
                  );

        }


        fprintf(fd,"%u\n",triModel->numberOfNormals );
        for (i=0; i<triModel->numberOfNormals; i++)
        {
          fprintf(
                   fd,"%f%f%f",
                   triModel->normal[(i*3)+0],
                   triModel->normal[(i*3)+1],
                   triModel->normal[(i*3)+2]
                  );

        }
        fprintf(fd,"end\n");
        fclose(fd);
        return 1;
    }
  return 0;
}



int saveModelTriHeader(const char * filename , struct TRI_Model * triModel)
{
  unsigned int i=0;
  FILE *fd=0;
  fd = fopen(filename,"w");
  if (fd!=0)
    {

        fprintf(fd,"const float %sVertices[] = { \n",filename);
        for (i=0; i<triModel->numberOfTriangles; i++)
        {
          fprintf(
                   fd,"%0.4f , %0.4f , %0.4f ",
                   triModel->triangleVertex[(i*3)+0],
                   triModel->triangleVertex[(i*3)+1],
                   triModel->triangleVertex[(i*3)+2]
                  );

          if (i+1<triModel->numberOfTriangles) { fprintf(fd,", \n"); }
        }
        fprintf(fd,"}; \n\n");


        fprintf(fd,"const float %sNormals[] = { ",filename);
        for (i=0; i<triModel->numberOfNormals; i++)
        {
           fprintf(
                   fd,"%0.4f , %0.4f , %0.4f , \n",
                   triModel->normal[(i*3)+0],
                   triModel->normal[(i*3)+1],
                   triModel->normal[(i*3)+2]
                  );
          if (i+1<triModel->numberOfNormals) { fprintf(fd,", \n"); }
        }
        fprintf(fd,"}; \n\n");

       fflush(fd);
       fclose(fd);
       return 1;
    }
  return 0;
}





