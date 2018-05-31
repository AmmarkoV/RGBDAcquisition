#include "ogl_rendering.h"


#include "FixedPipeline/ogl_fixed_pipeline_renderer.h"
#include "ShaderPipeline/ogl_shader_pipeline_renderer.h"

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../tools.h"


struct rendererConfiguration rendererOptions={0};


int resetRendererOptions()
{
  rendererOptions.doCulling  = 1;
  rendererOptions.useShaders = 0;
  return 1;
}


int enableShaders(char * vertShaderFilename , char * fragShaderFilename)
{
  strncpy(
           rendererOptions.fragmentShaderFile ,
           fragShaderFilename,
           MAX_SHADER_FILENAMES
          );
  rendererOptions.selectedFragmentShader = rendererOptions.fragmentShaderFile;

  strncpy(
           rendererOptions.vertexShaderFile ,
           vertShaderFilename,
           MAX_SHADER_FILENAMES
          );
  rendererOptions.selectedVertexShader = rendererOptions.vertexShaderFile;

  rendererOptions.useShaders=1;

  startOGLShaderPipeline(&rendererOptions);

  return 1;
}


int startOGLRendering()
{
  switch (rendererOptions.useShaders)
  {
    case 0 :
             startFixedOGLRendering(&rendererOptions);
    break;

    case 1 :
             fprintf(stderr,"enableShaders has already started the OGL Shader pipeline\n");
             //startShaderOGLRendering(&rendererOptions);
    break;
  };
 return 1;
}




int renderOGLBones( float * pos , unsigned int * parentNode ,  unsigned int boneSizes)
{
  switch (rendererOptions.useShaders)
  {
    case 0 :
         doOGLFixedBoneDrawCalllist(pos,parentNode,boneSizes);
    break;

    case 1 :
         doOGLShaderBoneDrawCalllist(pos,parentNode,boneSizes);
    break;
  };
  return 1;
}


int renderOGL(
               float * vertices ,       unsigned int numberOfVertices ,
               float * normal ,         unsigned int numberOfNormals ,
               float * textureCoords ,  unsigned int numberOfTextureCoords ,
               float * colors ,         unsigned int numberOfColors ,
               unsigned int * indices , unsigned int numberOfIndices
             )
{

  switch (rendererOptions.useShaders)
  {
    case 0 :
     doOGLGenericDrawCalllist(
                              vertices ,       numberOfVertices ,
                              normal ,         numberOfNormals ,
                              textureCoords ,  numberOfTextureCoords ,
                              colors ,         numberOfColors ,
                              indices ,        numberOfIndices
                             );
    break;

    case 1 :
      doOGLShaderDrawCalllist(
                              vertices ,       numberOfVertices ,
                              normal ,         numberOfNormals ,
                              textureCoords ,  numberOfTextureCoords ,
                              colors ,         numberOfColors ,
                              indices ,        numberOfIndices
                             );
    break;

  };

  return 1;
}




int stopOGLRendering()
{
switch (rendererOptions.useShaders)
  {
    case 0 :
     return stopOGLFixedRendering(&rendererOptions);
    break;

    case 1 :
     return stopOGLShaderRendering(&rendererOptions);
    break;
  };


 return 0;
}

