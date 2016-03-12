#include "model_loader_tri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if HAVE_OBJ_CODE_AVAILIABLE
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj)
{
   if (tri==0) { return 0; }
   if (obj==0) { return 0; }

   unsigned int i=0,j=0,pos=0;

       tri->header.triType=1;
       tri->header.numberOfTriangles = obj->numGroups * obj->numFaces * 3 ;
       tri->header.numberOfNormals   = obj->numGroups * obj->numFaces * 3 ;

       tri->triangleVertex = malloc(sizeof(float) * 3 * tri->header.numberOfTriangles);
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
        tri->normal = malloc(sizeof(float) * 3 * tri->header.numberOfNormals);
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
#endif // HAVE_OBJ_CODE_AVAILIABLE

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

        fprintf(stderr,"Reading %u bytes of vertex\n",sizeof(float) * 3 *triModel->header.numberOfTriangles);
        triModel->triangleVertex = ( float * ) malloc ( sizeof(float) * 3 * triModel->header.numberOfTriangles );
        n = fread(triModel->triangleVertex , sizeof(float), 3 * triModel->header.numberOfTriangles , fd);

        fprintf(stderr,"Reading %u bytes of normal\n",sizeof(float) * 3 * triModel->header.numberOfNormals);
        triModel->normal = ( float * ) malloc ( sizeof(float) * 3 * triModel->header.numberOfNormals );
        n = fread(triModel->normal , sizeof(float), 3 * triModel->header.numberOfNormals , fd);

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
        triModel->header.triType =2;
        triModel->header.floatSize =(unsigned int ) sizeof(float);
        fwrite (&triModel->header        , sizeof(struct TRI_Header), 1 , fd);
        fprintf(stderr,"Writing %u bytes of vertex\n", sizeof(float) * 3 * triModel->header.numberOfTriangles);
        fwrite (triModel->triangleVertex , 3*sizeof(float), triModel->header.numberOfTriangles, fd);
        fprintf(stderr,"Writing %u bytes of normal\n",sizeof(float) * 3 * triModel->header.numberOfNormals);
        fwrite (triModel->normal         , 3*sizeof(float), triModel->header.numberOfNormals  , fd);

        fflush(fd);
        fclose(fd);
        return 1;
    }
  return 0;
}



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
        for (i=0; i<triModel->header.numberOfTriangles; i++)
        {
          fprintf(
                   fd,"%0.4f , %0.4f , %0.4f ",
                   triModel->triangleVertex[(i*3)+0],
                   triModel->triangleVertex[(i*3)+1],
                   triModel->triangleVertex[(i*3)+2]
                  );

          if (i+1<triModel->header.numberOfTriangles) { fprintf(fd,", \n"); }
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





