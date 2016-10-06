#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>


#include "ogl_shader_pipeline_renderer.h"


void doOGLShaderDrawCalllist(
                              float * vertices ,       unsigned int numberOfVertices ,
                              float * normal ,         unsigned int numberOfNormals ,
                              float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              float * colors ,         unsigned int numberOfColors ,
                              unsigned int * indices , unsigned int numberOfIndices
                             )
{
  unsigned int i=0,z=0;


  glBegin(GL_TRIANGLES);
    if (numberOfIndices > 0 )
    {
     //fprintf(stderr,MAGENTA "drawing indexed TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
     unsigned int faceTriA,faceTriB,faceTriC,faceTriA_X,faceTriA_Y,faceTriA_Z,faceTriB_X,faceTriB_Y,faceTriB_Z,faceTriC_X,faceTriC_Y,faceTriC_Z;

     for (i = 0; i < numberOfIndices/3; i++)
     {
      faceTriA = indices[(i*3)+0];      faceTriB = indices[(i*3)+1];      faceTriC = indices[(i*3)+2];
      faceTriA_X = (faceTriA*3)+0;           faceTriA_Y = (faceTriA*3)+1;           faceTriA_Z = (faceTriA*3)+2;
      faceTriB_X = (faceTriB*3)+0;           faceTriB_Y = (faceTriB*3)+1;           faceTriB_Z = (faceTriB*3)+2;
      faceTriC_X = (faceTriC*3)+0;           faceTriC_Y = (faceTriC*3)+1;           faceTriC_Z = (faceTriC*3)+2;

      if (normal)   { glNormal3f(normal[faceTriA_X],normal[faceTriA_Y],normal[faceTriA_Z]); }
      if ( colors ) { glColor3f(colors[faceTriA_X],colors[faceTriA_Y],colors[faceTriA_Z]);  }
      glVertex3f(vertices[faceTriA_X],vertices[faceTriA_Y],vertices[faceTriA_Z]);

      if (normal)   { glNormal3f(normal[faceTriB_X],normal[faceTriB_Y],normal[faceTriB_Z]); }
      if ( colors ) { glColor3f(colors[faceTriB_X],colors[faceTriB_Y],colors[faceTriB_Z]);  }
      glVertex3f(vertices[faceTriB_X],vertices[faceTriB_Y],vertices[faceTriB_Z]);

      if (normal)   { glNormal3f(normal[faceTriC_X],normal[faceTriC_Y],normal[faceTriC_Z]); }
      if ( colors ) { glColor3f(colors[faceTriC_X],colors[faceTriC_Y],colors[faceTriC_Z]);  }
      glVertex3f(vertices[faceTriC_X],vertices[faceTriC_Y],vertices[faceTriC_Z]);
	 }
    } else
    {
      //fprintf(stderr,BLUE "drawing flat TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
      for (i=0; i<numberOfVertices/3; i++)
        {
         z=(i*3)*3;
         if (normal) { glNormal3f(normal[z+0],normal[z+1],normal[z+2]); }
         if (colors) { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);

         z+=3;
         if (normal) { glNormal3f(normal[z+0],normal[z+1],normal[z+2]); }
         if (colors) { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);
         z+=3;
         if (normal) { glNormal3f(normal[z+0],normal[z+1],normal[z+2]); }
         if (colors) { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);
        }
    }
  glEnd();
}
