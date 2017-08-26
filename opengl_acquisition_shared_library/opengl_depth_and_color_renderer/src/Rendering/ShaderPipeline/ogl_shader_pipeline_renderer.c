
#if USE_GLEW
#include <GL/glew.h>


// Here we decide which of the two versions we want to use
// If your systems supports both, choose to uncomment USE_OPENGL32
// otherwise choose to uncomment USE_OPENGL21
// GLView cna also help you decide before running this program:
//
// FOR MACOSX only, please use OPENGL32 for AntTweakBar to work properly
//
#define USE_OPENGL32


#define USE_SIMPLE_SHADERS 0


// GLFW lib
// http://www.glfw.org/documentation.html
#ifdef USE_OPENGL32
    #ifndef _WIN32
        #define GLFW_INCLUDE_GL3
        #define USE_GL3
        #define GLFW_NO_GLU
        #define GL3_PROTOTYPES 1
    #endif
#endif

#endif



#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>
#include <GL/glut.h>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ogl_shader_pipeline_renderer.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */





void doOGLShaderDrawCalllist(
                              float * vertices ,       unsigned int numberOfVertices ,
                              float * normal ,         unsigned int numberOfNormals ,
                              float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              float * colors ,         unsigned int numberOfColors ,
                              unsigned int * indices , unsigned int numberOfIndices
                             )
{
/*


void pushObjectToBufferData(
                             unsigned int * verticeCount ,
                             const float * vertices , unsigned int verticesLength ,
                             const float * normals , unsigned int normalsLength ,
                             const float * colors , unsigned int colorsLength ,
                             const float * texcoords , unsigned int texCoordsLength ,
                             int generateNewBuffer ,
                             GLuint buffer
                           )
{*/

#if USE_GLEW

    glBindBuffer( GL_ARRAY_BUFFER, buffer );        checkOpenGLError(__FILE__, __LINE__);



    *verticeCount+=(unsigned int ) verticesLength/(3*sizeof(float));
    fprintf(stderr,GREEN "Will DrawArray(GL_TRIANGLES,0,%u) - %u \n" NORMAL ,*verticeCount,verticesLength);
    fprintf(stderr,GREEN "Pushing %u vertices (%u bytes) and %u normals (%u bytes) and %u colors and %u texture coords as our object \n" NORMAL ,verticesLength/sizeof(float),verticesLength,normalsLength/sizeof(float),normalsLength,colorsLength,texCoordsLength);
  if (generateNewBuffer)
   {
    glBufferData( GL_ARRAY_BUFFER, verticesLength + normalsLength  + colorsLength + texCoordsLength ,NULL, GL_STREAM_DRAW ); checkOpenGLError(__FILE__, __LINE__);

    glBufferSubData( GL_ARRAY_BUFFER, 0                                      , verticesLength , vertices );                  checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, verticesLength                         , normalsLength  , normals );                   checkOpenGLError(__FILE__, __LINE__);

    if ( (colors!=0) && (colorsLength!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, verticesLength + normalsLength , colorsLength , colors );                     checkOpenGLError(__FILE__, __LINE__);
    }
    if ( (texcoords!=0) && (texCoordsLength!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, verticesLength + normalsLength + colorsLength, texCoordsLength , texcoords ); checkOpenGLError(__FILE__, __LINE__);
    }
   }


    vPosition = glGetAttribLocation( program, "vPosition" );                                   checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vPosition );                                                    checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(0) );             checkOpenGLError(__FILE__, __LINE__);

     vNormal = glGetAttribLocation( program, "vNormal" );                                      checkOpenGLError(__FILE__, __LINE__);
     glEnableVertexAttribArray( vNormal );                                                     checkOpenGLError(__FILE__, __LINE__);
     glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(verticesLength) ); checkOpenGLError(__FILE__, __LINE__);


    if ( (colors!=0) && (colorsLength!=0) )
    {
     vColor = glGetAttribLocation( program, "vColor" );
     glEnableVertexAttribArray( vColor );
     glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( verticesLength + normalsLength ) );
     checkOpenGLError(__FILE__, __LINE__);
    }


    textureStrengthLocation = glGetUniformLocation(program, "textureStrength");  checkOpenGLError(__FILE__, __LINE__);
    if ( (texcoords!=0) && (texCoordsLength!=0) )
    {
     vTexture = glGetAttribLocation( program, "vTexture" );
     glEnableVertexAttribArray( vTexture );
     glVertexAttribPointer( vTexture, 2, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( verticesLength + normalsLength + colorsLength) );
     checkOpenGLError(__FILE__, __LINE__);

     //textureStrength[0]=1.0;
    } else
    { textureStrength[0]=0.0; }



#endif // USE_GLEW


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
