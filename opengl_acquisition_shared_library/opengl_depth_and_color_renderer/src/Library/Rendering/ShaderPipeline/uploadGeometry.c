#include <stdio.h>

#if USE_GLEW
// Include GLEW
#include <GL/glew.h>
#endif // USE_GLEW


#include "uploadGeometry.h"


//#include <GL/gl.h>  //Also on header..
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>


#define BUFFER_OFFSET( offset )   ((GLvoid*) (offset))

GLuint
pushObjectToBufferData(
                             int generateNewVao,
                             GLuint *vao ,
                             GLuint *arrayBuffer ,
                             GLuint programID  ,
                             const float * vertices , unsigned int sizeOfVertices ,
                             const float * normals , unsigned int sizeOfNormals ,
                             const float * textureCoords ,  unsigned int sizeOfTextureCoords ,
                             const float * colors , unsigned int sizeOfColors,
                             const unsigned int * indices , unsigned int sizeOfIndices
                           )
{
    #if USE_GLEW
    if (generateNewVao)
    {
      glGenVertexArrays(1, vao);     checkOpenGLError(__FILE__, __LINE__);
    }

    glBindVertexArray(*vao); checkOpenGLError(__FILE__, __LINE__);


    // Create and initialize a buffer object on the server side (GPU)
    glGenBuffers( 1, arrayBuffer );                   checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, *arrayBuffer );    checkOpenGLError(__FILE__, __LINE__);

    unsigned int NumVertices=(unsigned int ) sizeOfVertices/(3*sizeof(float));
    fprintf(stderr,"Will DrawArray(GL_TRIANGLES,0,%u) - %u \n"  ,NumVertices,sizeOfVertices);

    fprintf(stderr,
             "Pushing %lu vertices (%u bytes) and %u normals (%u bytes) as our object \n"  ,
             (unsigned long) sizeOfVertices/sizeof(float),
             sizeOfVertices,
             sizeOfNormals/sizeof(float),
             sizeOfNormals
            );

    glBufferData(GL_ARRAY_BUFFER,sizeOfVertices+sizeOfNormals+sizeOfColors+sizeOfTextureCoords,NULL,GL_STATIC_DRAW);    checkOpenGLError(__FILE__, __LINE__);

    GLintptr memoryOffset=0;
    if ((vertices!=0) && (sizeOfVertices!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfVertices, vertices);               checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfVertices;
    }

    if ( (normals!=0) && (sizeOfNormals!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfNormals, normals);                checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfNormals;
    }

    if ( (colors!=0) && (sizeOfColors!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfColors, colors);                   checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfColors;
    }

    if ( (textureCoords!=0) && (sizeOfTextureCoords!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, memoryOffset, sizeOfTextureCoords , textureCoords );  checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfTextureCoords;
    }


    memoryOffset=0;
    if ((vertices!=0) && (sizeOfVertices!=0) )
    {
     //Pass vPosition to shader
     GLuint vPosition = glGetAttribLocation( programID, "vPosition" );                         checkOpenGLError(__FILE__, __LINE__);
     if (GL_INVALID_OPERATION != vPosition)
     {
      glEnableVertexAttribArray(vPosition);                                                    checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(memoryOffset) ); checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfVertices;
     }
    }

    if ((normals!=0) && (sizeOfNormals!=0) )
    {
     //Pass vNormal to shader
     GLuint vNormal = glGetAttribLocation(programID, "vNormal");                             checkOpenGLError(__FILE__, __LINE__);
     if (GL_INVALID_OPERATION != vNormal)
     {
      glEnableVertexAttribArray(vNormal);                                                    checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vNormal,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));      checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfNormals;
     }
    }

    if ( (colors!=0) && (sizeOfColors!=0) )
    {
     //Pass vColor to shader
     GLuint vColor = glGetAttribLocation(programID, "vColor");                               checkOpenGLError(__FILE__, __LINE__);
     if (GL_INVALID_OPERATION != vColor)
     {
      glEnableVertexAttribArray(vColor);                                                     checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vColor,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));       checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfColors;
     }
    }

    //GLuint textureStrengthLocation = glGetUniformLocation(programID, "textureStrength");  checkOpenGLError(__FILE__, __LINE__);
    if ( (textureCoords!=0) && (sizeOfTextureCoords!=0) )
    {
     //Pass vTexture to shader
     GLuint vTexture = glGetAttribLocation(programID, "vTexture");                           checkOpenGLError(__FILE__, __LINE__);
     if (GL_INVALID_OPERATION != vTexture )
     {
      glEnableVertexAttribArray(vTexture);                                                   checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vTexture,2,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));     checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfTextureCoords;
     }
    }

   return 1;
  #else
   #warning "USE_GLEW not defined, uploadGeometry will not be included.."
   return 0;
  #endif
}
