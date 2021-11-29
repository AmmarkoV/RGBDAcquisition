#include "ogl_rendering.h"


#include "FixedPipeline/ogl_fixed_pipeline_renderer.h"
#include "ShaderPipeline/ogl_shader_pipeline_renderer.h"
#include "ShaderPipeline/shader_loader.h"

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../Tools/tools.h"


struct rendererConfiguration rendererOptions={0};


int resetRendererOptions()
{
  rendererOptions.doCulling  = 1;
  //rendererOptions.useShaders = 0;
  return 1;
}


int enableShaders(const char * vertShaderFilename ,const char * fragShaderFilename)
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

 // #warning "Decide startOGLShaderPipeline or startShaderOGLRendering"
  startOGLShaderPipeline(&rendererOptions);
  //TODO: startShaderOGLRendering


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



int renderOGLLight( float * pos , unsigned int * parentNode ,  unsigned int boneSizes)
{
  rendererOptions.useLighting=1; //if we want light we need to use lighting..
  rendererOptions.lightPos[0]=pos[0];
  rendererOptions.lightPos[1]=pos[1];
  rendererOptions.lightPos[2]=pos[2];

  switch (rendererOptions.useShaders)
  {
    case 0 :
          fixedOGLLighting(&rendererOptions);
    break;

    case 1 :
         fprintf(stderr,"No Shader lighting support yet..!\n");
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
               const float * projectionMatrix ,
               const float * viewMatrix ,
               const float * modelMatrix ,
               const float * mvpMatrix ,
               //-------------------------------------------------------
               const float * vertices ,       unsigned int numberOfVertices ,
               const float * normal ,         unsigned int numberOfNormals ,
               const float * textureCoords ,  unsigned int numberOfTextureCoords ,
               const float * colors ,         unsigned int numberOfColors ,
               const unsigned int * indices , unsigned int numberOfIndices
             )
{

  switch (rendererOptions.useShaders)
  {
    case 0 :
     doOGLGenericDrawCalllist(
                              projectionMatrix ,
                              viewMatrix ,
                              modelMatrix ,
                              mvpMatrix ,
                              //-------------------------------------------------------
                              vertices ,       numberOfVertices ,
                              normal ,         numberOfNormals ,
                              textureCoords ,  numberOfTextureCoords ,
                              colors ,         numberOfColors ,
                              indices ,        numberOfIndices
                             );
    break;

    case 1 :
      fprintf(stderr,"shader draw is under construction please deactivate shaders from CMake..\n");
      fprintf(stderr,"renderOGL => rendererOptions.useShaders=%d\n",rendererOptions.useShaders);
      useShader(rendererOptions.loadedShader);
      doOGLShaderDrawCalllist(
                              projectionMatrix ,
                              viewMatrix ,
                              modelMatrix ,
                              mvpMatrix ,
                              //-------------------------------------------------------
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

