#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>
#include <math.h>


#include "ogl_fixed_pipeline_renderer.h"

#define boneSphere 0.05

void doOGLBoneDrawCalllist( float * pos , unsigned int * parentNode ,  unsigned int boneSizes)
{
  unsigned int bone=0;


  glLineWidth(6.0);
  for (bone=0; bone<boneSizes; bone++)
  {
     unsigned int parentBone = parentNode[bone];

     if (parentBone!=bone)
     {
       if (parentBone<boneSizes)
       {
        glBegin(GL_LINES);
         glColor3f(0.4,0.01,0.0);
         glVertex3f(pos[parentBone*3+0],pos[parentBone*3+1],pos[parentBone*3+2]);
         glColor3f(0.4,0.01,0.0);
         glVertex3f(pos[bone*3+0],pos[bone*3+1],pos[bone*3+2]);
        glEnd();
       }
     }
  }
  glLineWidth(1.0);

  for (bone=0; bone<boneSizes; bone++)
  {

   if ( (pos[bone*3+0]!=pos[bone*3+0]) ||
        (pos[bone*3+1]!=pos[bone*3+1]) ||
        (pos[bone*3+2]!=pos[bone*3+2])
       )
       {

       } else
       {


     int quality=20;
//    double r=1.0;
    int lats=quality;
    int longs=quality;
  //---------------
    int i, j;
    for(i = 0; i <= lats; i++)
    {
       double lat0 = M_PI * (-0.5 + (double) (i - 1) / lats);
       double z0  = sin(lat0);
       double zr0 =  cos(lat0);

       double lat1 = M_PI * (-0.5 + (double) i / lats);
       double z1 = sin(lat1);
       double zr1 = cos(lat1);

 glPushMatrix();
  glTranslatef(pos[bone*3+0],pos[bone*3+1],pos[bone*3+2]);
       glScalef( boneSphere , boneSphere , boneSphere );
       glBegin(GL_QUAD_STRIP);
       glColor3f(0.74,0.01,1.0);
       for(j = 0; j <= longs; j++)
        {
           double lng = 2 * M_PI * (double) (j - 1) / longs;
           double x = cos(lng);
           double y = sin(lng);

           glNormal3f(x * zr0, y * zr0, z0);
           glVertex3f(x * zr0, y * zr0, z0);
           glNormal3f(x * zr1, y * zr1, z1);
           glVertex3f(x * zr1, y * zr1, z1);
        }
       glEnd();

 glTranslatef(-pos[bone*3+0],-pos[bone*3+1],-pos[bone*3+2]);
glPopMatrix();



       }
   }







  }



/*

*/

}


void doOGLGenericDrawCalllist(
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
      faceTriA = indices[(i*3)+0];           faceTriB = indices[(i*3)+1];           faceTriC = indices[(i*3)+2];
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
