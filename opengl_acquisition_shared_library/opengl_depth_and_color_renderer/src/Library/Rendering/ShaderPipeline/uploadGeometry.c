#include <stdio.h>

#if USE_GLEW
// Include GLEW
#include <GL/glew.h>
#endif // USE_GLEW


#include "uploadGeometry.h"


//#include <GL/gl.h>  //Also on header..
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include "../../Tools/tools.h"

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

    glBindVertexArray(*vao);         checkOpenGLError(__FILE__, __LINE__);

    // Create and initialize a buffer object on the server side (GPU)
    glGenBuffers( 1, arrayBuffer );                   checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, *arrayBuffer );    checkOpenGLError(__FILE__, __LINE__);

    unsigned int numVertices=(unsigned int ) sizeOfVertices/(3*sizeof(float));
    fprintf(stderr,"Will DrawArray(GL_TRIANGLES,0,%u) - %u \n",numVertices,sizeOfVertices);
    fprintf(stderr,"Pushing %lu vertices (%u bytes) and %lu normals (%u bytes) as our object \n"  ,
            (unsigned long) sizeOfVertices/sizeof(float),
            sizeOfVertices,
            sizeOfNormals/sizeof(float),
            sizeOfNormals
           );

    //If no data given, zero their size..
    if (vertices==0)      { sizeOfVertices=0;      }
    if (normals==0)       { sizeOfNormals=0;       }
    if (textureCoords==0) { sizeOfTextureCoords=0; }
    if (colors==0)        { sizeOfColors=0;        }
    if (indices==0)       { sizeOfIndices=0;       }

    GLintptr memoryOffset=0;
    GLsizeiptr totalBufferDataSize=sizeOfVertices+sizeOfNormals+sizeOfColors+sizeOfTextureCoords;
    //----------------------------------------------------------------------------------------------------------------------------
    glBufferData(GL_ARRAY_BUFFER,totalBufferDataSize,NULL,GL_STATIC_DRAW);                   checkOpenGLError(__FILE__, __LINE__);
    //----------------------------------------------------------------------------------------------------------------------------
    if ((vertices!=0) && (sizeOfVertices!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfVertices, vertices);               checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfVertices;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (colors!=0) && (sizeOfColors!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfColors, colors);                   checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfColors;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (normals!=0) && (sizeOfNormals!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfNormals, normals);                 checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfNormals;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (textureCoords!=0) && (sizeOfTextureCoords!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, memoryOffset, sizeOfTextureCoords , textureCoords );  checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfTextureCoords;
    }
    //----------------------------------------------------------------------------------------------------------------------------


    memoryOffset=0;
    //----------------------------------------------------------------------------------------------------------------------------
    if ((vertices!=0) && (sizeOfVertices!=0))
    {
     //Pass vPosition to shader
     GLuint vPosition = glGetAttribLocation(programID, "vPosition" );                       checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vPosition) && (vPosition!=-1) )
     {
      glEnableVertexAttribArray(vPosition);                                                  checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vPosition,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));    checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfVertices;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (colors!=0) && (sizeOfColors!=0) )
    {
     //Pass vColor to shader
     GLuint vColor = glGetAttribLocation(programID, "vColor");                               checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vColor) && (vColor!=-1) )
     {
      glEnableVertexAttribArray(vColor);                                                     checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vColor,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));       checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfColors;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ((normals!=0) && (sizeOfNormals!=0))
    {
     //Pass vNormal to shader
     GLuint vNormal = glGetAttribLocation(programID, "vNormal");                             checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vNormal) && (vNormal!=-1) )
     {
      glEnableVertexAttribArray(vNormal);                                                    checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vNormal,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));      checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfNormals;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (textureCoords!=0) && (sizeOfTextureCoords!=0) )
    {
     //Pass vTexture to shader
     GLuint vTexture = glGetAttribLocation(programID, "vTexture");                           checkOpenGLError(__FILE__, __LINE__);
    //GLuint textureStrengthLocation = glGetUniformLocation(programID, "textureStrength");  checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vTexture ) && (vTexture!=-1) )
     {
      glEnableVertexAttribArray(vTexture);                                                   checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vTexture,2,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(memoryOffset));     checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfTextureCoords;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------

   return 1;
  #else
   #warning "USE_GLEW not defined, uploadGeometry will not be included.."
   return 0;
  #endif
}
