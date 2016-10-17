#define USE_GLEW 0


#if USE_GLEW
#include <GL/glew.h>
#endif

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ogl_shader_pipeline_renderer.h"


void doOGLShaderDrawCalllist(
                              float * vertices ,       unsigned int numberOfVertices ,
                              float * normal ,         unsigned int numberOfNormals ,
                              float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              float * colors ,         unsigned int numberOfColors ,
                              unsigned int * indices , unsigned int numberOfIndices
                             )
{
  fprintf(stderr,"OGLShader Draw list not implemented..!\n");
}





int startOGLShaderPipeline()
{
#if USE_GLEW
  glewInit();
  fprintf(stderr,"Using GLEW %s \n",glewGetString(GLEW_VERSION));

  if (GLEW_VERSION_3_2)
    {
      fprintf(stderr,"Yay! OpenGL 3.2 is supported and GLSL 1.5!\n");
    }

	if (glewIsSupported("GL_ARB_vertex_buffer_object"))   { fprintf(stderr,"ARB VBO's are supported\n");  } else
    if (glewIsSupported("GL_APPLE_vertex_buffer_object")) { fprintf(stderr,"APPLE VBO's are supported\n");} else
		                                                  { fprintf(stderr,"VBO's are not supported,program will not run!!!\n"); }


	if (glewIsSupported("GL_ARB_vertex_array_object"))    { fprintf(stderr,"ARB VAO's are supported\n"); } else
	//this is the name of the extension for GL2.1 in MacOSX
    if (glewIsSupported("GL_APPLE_vertex_array_object"))  { fprintf(stderr,"APPLE VAO's are supported\n"); } else
		                                                  { fprintf(stderr,"VAO's are not supported, program will not run!!!\n"); }


    fprintf(stderr,"Vendor: %s \n",glGetString (GL_VENDOR) );
    fprintf(stderr,"Renderer: %s \n",glGetString (GL_RENDERER) );
    fprintf(stderr,"Version: %s \n",glGetString (GL_VERSION) );
#endif


}
