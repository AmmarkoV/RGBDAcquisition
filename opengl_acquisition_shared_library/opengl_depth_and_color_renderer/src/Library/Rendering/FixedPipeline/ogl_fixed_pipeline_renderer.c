#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ogl_fixed_pipeline_renderer.h"
#include "../../Scene/scene.h"
#include "../../Tools/tools.h"

#define boneSphere 0.05




int  fixedOGLLighting(struct rendererConfiguration * config)
{ 
  //#warning "GL_COLOR does not even exist"
  //glEnable(GL_COLOR);
  //if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color \n"); }
  glEnable(GL_COLOR_MATERIAL);
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color material\n"); }

  #if USE_LIGHTS
   if (config->useLighting)
   {
      GLfloat light_position[] = { config->lightPos[0], config->lightPos[1], config->lightPos[2] , 1.0 };
      GLfloat light_ambient[] = { 1.0, 1.0, 1.0, 1.0 };
      GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
      GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0 };
  
      GLfloat mat_ambient[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat mat_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat mat_shininess[] = { 50.0 };
  
  
      glEnable(GL_LIGHT0);
      glEnable(GL_LIGHTING);
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling lighting\n"); }
      glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
      glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
      glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
      glLightfv(GL_LIGHT0, GL_POSITION, light_position);
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up lights\n"); }

      GLenum faces=GL_FRONT; //GL_FRONT_AND_BACK;//
      glMaterialfv(faces, GL_AMBIENT,    mat_ambient);
      glMaterialfv(faces, GL_DIFFUSE,    mat_diffuse);
      glMaterialfv(faces, GL_SPECULAR,   mat_specular);
      glMaterialfv(faces, GL_SHININESS,   mat_shininess); // <- this was glMateriali
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up Front/Back lights\n"); }  
      return 1;
   }
  #else
   fprintf(stderr,"Please note that lighting is disabled via the USE_LIGHTS precompiler define\n");
   return 0; 
  #endif // USE_LIGHTS
}


int startFixedOGLRendering(struct rendererConfiguration * config)
{
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error while initializing scene\n"); }
  glEnable(GL_DEPTH_TEST); /* enable depth buffering */
  glDepthFunc(GL_LESS);    /* pedantic, GL_LESS is the default */
  glDepthMask(GL_TRUE);
  glClearDepth(1.0);       /* pedantic, 1.0 is the default */

  //HQ settings
  glEnable(GL_NORMALIZE);
  glShadeModel(GL_SMOOTH);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error while initializing HQ settings\n"); }

  /* frame buffer clears should be to black */
  glClearColor(0.0, 0.0, 0.0, 0.0);

  /* set up projection transform */
  glMatrixMode(GL_PROJECTION);

  updateProjectionMatrix();
  if (checkOpenGLError(__FILE__, __LINE__))
     { fprintf(stderr,"OpenGL error after updating projection matrix\n"); }

  /* establish initial viewport */
  /* pedantic, full window size is default viewport */

   fixedOGLLighting(config);


  //This is not needed -> :P  glCullFace(GL_FRONT_AND_BACK);
  //Enable Culling
  if (config->doCulling)
  {
   glFrontFace(GL_CCW); //GL_CW / GL_CCW
     if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glFrontFace(GL_CCW); \n"); }
   glCullFace(GL_BACK);
     if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glCullFace(GL_BACK); \n"); }
   glEnable(GL_CULL_FACE);
    if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glEnable(GL_CULL_FACE); \n"); }
  }
 return 1;
}


int stopOGLFixedRendering(struct rendererConfiguration * config)
{
  return 1;
}






void doOGLFixedBoneDrawCalllist(float * pos , unsigned int * parentNode ,  unsigned int boneSizes)
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
  // float r=1.0;
     int lats=quality;
     int longs=quality;
  //---------------
    int i, j;
    for(i = 0; i <= lats; i++)
    {
       float lat0 = M_PI * (-0.5 + (float) (i - 1) / lats);
       float z0  = sin(lat0);
       float zr0 =  cos(lat0);

       float lat1 = M_PI * (-0.5 + (float) i / lats);
       float z1 = sin(lat1);
       float zr1 = cos(lat1);

  //---------------
   glPushMatrix();
    glTranslatef(pos[bone*3+0],pos[bone*3+1],pos[bone*3+2]);
       glScalef(boneSphere,boneSphere,boneSphere);
       glBegin(GL_QUAD_STRIP);
       glColor3f(0.74,0.01,1.0);
       for(j = 0; j <= longs; j++)
        {
           float lng = 2 * M_PI * (float) (j - 1) / longs;
           float x = cos(lng);
           float y = sin(lng);

           glNormal3f(x * zr0, y * zr0, z0);
           glVertex3f(x * zr0, y * zr0, z0);
           glNormal3f(x * zr1, y * zr1, z1);
           glVertex3f(x * zr1, y * zr1, z1);
        }
       glEnd();

   glTranslatef(-pos[bone*3+0],-pos[bone*3+1],-pos[bone*3+2]);
  glPopMatrix();
  //---------------



       }
   }

  }
}


void doOGLGenericDrawCalllist(
                              const float * projectionMatrix ,
                              const float * viewMatrix ,
                              const float * modelMatrix ,
                              const float * mvpMatrix ,
                              //-------------------------------------------------------
                              const float * vertices ,       unsigned int numberOfVertices ,
                              const float * normals ,        unsigned int numberOfNormals ,
                              const float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              const float * colors ,         unsigned int numberOfColors ,
                              const unsigned int * indices , unsigned int numberOfIndices
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

      if (normals)   { glNormal3f(normals[faceTriA_X],normals[faceTriA_Y],normals[faceTriA_Z]); }
      if ( colors )  { glColor3f(colors[faceTriA_X],colors[faceTriA_Y],colors[faceTriA_Z]);  }
      glVertex3f(vertices[faceTriA_X],vertices[faceTriA_Y],vertices[faceTriA_Z]);

      if (normals)   { glNormal3f(normals[faceTriB_X],normals[faceTriB_Y],normals[faceTriB_Z]); }
      if ( colors )  { glColor3f(colors[faceTriB_X],colors[faceTriB_Y],colors[faceTriB_Z]);  }
      glVertex3f(vertices[faceTriB_X],vertices[faceTriB_Y],vertices[faceTriB_Z]);

      if (normals)   { glNormal3f(normals[faceTriC_X],normals[faceTriC_Y],normals[faceTriC_Z]); }
      if ( colors )  { glColor3f(colors[faceTriC_X],colors[faceTriC_Y],colors[faceTriC_Z]);  }
      glVertex3f(vertices[faceTriC_X],vertices[faceTriC_Y],vertices[faceTriC_Z]);
	 }
    } else
    {
      //fprintf(stderr,BLUE "drawing flat TRI\n" NORMAL); //dbg msg to be sure what draw operation happens here..!
      for (i=0; i<numberOfVertices/3; i++)
        {
         z=(i*3)*3;
         if (normals) { glNormal3f(normals[z+0],normals[z+1],normals[z+2]); }
         if (colors)  { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);

         z+=3;
         if (normals) { glNormal3f(normals[z+0],normals[z+1],normals[z+2]); }
         if (colors)  { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);
         z+=3;
         if (normals) { glNormal3f(normals[z+0],normals[z+1],normals[z+2]); }
         if (colors)  { glColor3f(colors[z+0],colors[z+1],colors[z+2]);  }
                            glVertex3f(vertices[z+0],vertices[z+1],vertices[z+2]);
        }
    }
  glEnd();
}
