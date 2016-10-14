#include "ogl_rendering.h"


#include "FixedPipeline/ogl_fixed_pipeline_renderer.h"
#include "ShaderPipeline/ogl_shader_pipeline_renderer.h"

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>



#define MAX_FILENAMES 512
//Shader specific stuff ----------------
char fragmentShaderFile[MAX_FILENAMES]={0};
char * selectedFragmentShader = 0;
char vertexShaderFile[MAX_FILENAMES]={0};
char * selectedVertexShader = 0;
struct shaderObject * loadedShader=0;
//--------------------------------------

int doCulling=1;

int enableShaders(char * vertShaderFilename , char * fragShaderFilename)
{
  strncpy(fragmentShaderFile , fragShaderFilename,MAX_FILENAMES);
  selectedFragmentShader = fragmentShaderFile;

  strncpy(vertexShaderFile , vertShaderFilename,MAX_FILENAMES);
  selectedVertexShader = vertexShaderFile;

  return 1;
}


int startOGLRendering()
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


  #warning "GL_COLOR does not even exist"
  //glEnable(GL_COLOR);
  //if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color \n"); }
  glEnable(GL_COLOR_MATERIAL);
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color material\n"); }

  #if USE_LIGHTS
   glEnable(GL_LIGHT0);
   glEnable(GL_LIGHTING);
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling lighting\n"); }
   glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
   glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
   glLightfv(GL_LIGHT0, GL_POSITION, light_position);
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up lights\n"); }

   GLenum faces=GL_FRONT;//GL_FRONT_AND_BACK;
   glMaterialfv(faces, GL_AMBIENT,    mat_ambient);
   glMaterialfv(faces, GL_DIFFUSE,    mat_diffuse);
   glMaterialfv(faces, GL_SPECULAR,   mat_specular);
   glMaterialfv(faces, GL_SHININESS,   mat_shininess); // <- this was glMateriali
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up Front/Back lights\n"); }
  #else
   fprintf(stderr,"Please note that lighting is disabled via the USE_LIGHTS precompiler define\n");
  #endif // USE_LIGHTS


  if ( ( selectedFragmentShader != 0) || ( selectedVertexShader != 0 ) )
  {
      loadedShader = loadShader(selectedVertexShader,selectedFragmentShader);
  }

  //This is not needed -> :P  glCullFace(GL_FRONT_AND_BACK);
  //Enable Culling
  if (doCulling)
  {
   glFrontFace(GL_CCW); //GL_CW / GL_CCW
     if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glFrontFace(GL_CCW); \n"); }
   glCullFace(GL_BACK);
     if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glCullFace(GL_BACK); \n"); }
   glEnable(GL_CULL_FACE);
    if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error glEnable(GL_CULL_FACE); \n"); }
  }

}



int renderOGL(
               float * vertices ,       unsigned int numberOfVertices ,
               float * normal ,         unsigned int numberOfNormals ,
               float * textureCoords ,  unsigned int numberOfTextureCoords ,
               float * colors ,         unsigned int numberOfColors ,
               unsigned int * indices , unsigned int numberOfIndices
             )
{

   doOGLGenericDrawCalllist(
                              vertices ,       numberOfVertices ,
                              normal ,         numberOfNormals ,
                              textureCoords ,  numberOfTextureCoords ,
                              colors ,         numberOfColors ,
                              indices ,        numberOfIndices
                             );
  return 1;
}




int stopOGLRendering()
{

  if ( ( selectedFragmentShader != 0) || ( selectedVertexShader != 0 ) )
  {
      unloadShader(loadedShader);
  }

}

