#include "ogl_fixed_pipeline_renderer.h"

 /*
void doTriDrawCalllist(
                         float * vertices ,       unsigned int numberOfVertices ,
                         float * normal ,         unsigned int numberOfNormals ,
                         float * textureCoords ,  unsigned int numberOfTextureCoords ,
                         float * colors ,         unsigned int numberOfColors ,
                         unsigned int * indices , unsigned int numberOfIndices
                     )
{
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
}
*/
