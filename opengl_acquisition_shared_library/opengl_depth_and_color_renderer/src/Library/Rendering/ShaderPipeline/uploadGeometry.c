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



GLuint
pushBonesToBufferData(
                        int generateNewVao,
                        int generateNewArrayBuffer,
                        int generateNewElementBuffer,
                        GLuint programID,
                        //-------------------------------------------------------------------
                        struct shaderModelData * shaderData
                     )
{
   #if USE_GLEW
    fprintf(stderr,"PushBones(BoneIndexes,0,%u) ",shaderData->sizeOfBoneIndexes);

    if (shaderData->numberOfBonesPerVertex>4)
    {
       fprintf(stderr,"Too many bones per vertex (%u) ",shaderData->numberOfBonesPerVertex);
       return 0;
    }

    //If no data given, zero their size..
    if (shaderData->boneIndexes==0)      { shaderData->sizeOfBoneIndexes=0;      }
    if (shaderData->boneWeightValues==0) { shaderData->sizeOfBoneWeightValues=0; }
    if (shaderData->boneTransforms==0)   { shaderData->sizeOfBoneTransforms=0;   }


    if (generateNewVao)
      { glGenVertexArrays(1,&shaderData->VAO);     checkOpenGLError(__FILE__, __LINE__); }
    glBindVertexArray(shaderData->VAO);            checkOpenGLError(__FILE__, __LINE__);
    //--------------------------------------------------------------------------

    if (generateNewElementBuffer)
      { glGenBuffers(1, &shaderData->elementBuffer);            checkOpenGLError(__FILE__, __LINE__); }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shaderData->elementBuffer);                                       checkOpenGLError(__FILE__, __LINE__);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, shaderData->sizeOfIndices, shaderData->indices , GL_STATIC_DRAW); checkOpenGLError(__FILE__, __LINE__);





    // Create and initialize a buffer object on the server side (GPU)
    if (generateNewArrayBuffer)
     { glGenBuffers( 1, &shaderData->arrayBuffer );                   checkOpenGLError(__FILE__, __LINE__); }
    glBindBuffer( GL_ARRAY_BUFFER, shaderData->arrayBuffer );         checkOpenGLError(__FILE__, __LINE__);




    //Create buffer data..
    GLsizei    stride = 0;
    GLintptr   memoryOffset=0;
    GLsizeiptr totalBufferDataSize=shaderData->sizeOfBoneIndexes+shaderData->sizeOfBoneWeightValues+shaderData->sizeOfBoneTransforms;
    //----------------------------------------------------------------------------------------------------------------------------
    glBufferData(GL_ARRAY_BUFFER,totalBufferDataSize,NULL,GL_STATIC_DRAW);                                  checkOpenGLError(__FILE__, __LINE__);
    //----------------------------------------------------------------------------------------------------------------------------
    if  (shaderData->sizeOfBoneIndexes!=0)
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfBoneIndexes, shaderData->boneIndexes);                        checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=shaderData->sizeOfBoneIndexes;
     //stride += 3 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfBoneWeightValues!=0)
    {
     glBufferSubData( GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfBoneWeightValues , shaderData->boneWeightValues );            checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=shaderData->sizeOfBoneWeightValues;
     //stride += 2 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfBoneTransforms!=0)
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfBoneTransforms, shaderData->boneTransforms);                   checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=shaderData->sizeOfBoneTransforms;
     //stride += 3 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------


    memoryOffset=0;

    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfBoneIndexes!=0)
    {
     //Pass vBoneIndexIDs to shader
     GLuint vBoneIndexIDs = glGetAttribLocation(programID, "vBoneIndexIDs" );                                             checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vBoneIndexIDs) && (vBoneIndexIDs!=-1) )
     {
      glEnableVertexAttribArray(vBoneIndexIDs);                                                                           checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vBoneIndexIDs,shaderData->numberOfBonesPerVertex,GL_UNSIGNED_INT,GL_FALSE,stride,(GLvoid*) memoryOffset); checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfBoneIndexes;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfBoneWeightValues!=0)
    {
     //Pass vBoneWeightValues to shader
     GLuint vBoneWeightValues = glGetAttribLocation(programID, "vBoneWeightValues");                                      checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vBoneWeightValues ) && (vBoneWeightValues!=-1) )
     {
      glEnableVertexAttribArray(vBoneWeightValues);                                                                       checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vBoneWeightValues,shaderData->numberOfBonesPerVertex,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);    checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfBoneWeightValues;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if  (shaderData->sizeOfBoneTransforms!=0)
    {
     //Pass vBoneTransform to shader
     GLuint vBoneTransform = glGetAttribLocation(programID, "vBoneTransform");                        checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vBoneTransform) && (vBoneTransform!=-1) )
     {
      glEnableVertexAttribArray(vBoneTransform);                                                      checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vBoneTransform,16,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);       checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfBoneTransforms;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------

   return 1;
  #else
   #warning "USE_GLEW not defined, pushBonesToBufferData will not be included.."
   return 0;
  #endif
}




GLuint
pushObjectToBufferData(
                        int generateNewVao,
                        GLuint *vao ,
                        GLuint *arrayBuffer,
                        GLuint *elementBuffer,
                        GLuint programID  ,
                        const float * vertices , unsigned int sizeOfVertices ,
                        const float * normals , unsigned int sizeOfNormals ,
                        const float * textureCoords ,  unsigned int sizeOfTextureCoords ,
                        const float * colors , unsigned int sizeOfColors,
                        const unsigned int * indices , unsigned int sizeOfIndices
                      )
{
    #if USE_GLEW

    unsigned int numVertices=(unsigned int ) sizeOfVertices/(3*sizeof(float));
    fprintf(stderr,"DrawArray(GL_TRIANGLES,0,%u) ",numVertices);
    fprintf(stderr,"%lu vertices (%u bytes) and %lu normals (%u bytes)\n"  ,
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


    if (generateNewVao)
      { glGenVertexArrays(1, vao);     checkOpenGLError(__FILE__, __LINE__); }
    glBindVertexArray(*vao);           checkOpenGLError(__FILE__, __LINE__);
    //--------------------------------------------------------------------------


    if (indices!=0)
    {
      glGenBuffers(1, elementBuffer);                                                     checkOpenGLError(__FILE__, __LINE__);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *elementBuffer);                              checkOpenGLError(__FILE__, __LINE__);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeOfIndices, indices , GL_STATIC_DRAW);     checkOpenGLError(__FILE__, __LINE__);
    }




    // Create and initialize a buffer object on the server side (GPU)
    glGenBuffers( 1, arrayBuffer );                   checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, *arrayBuffer );    checkOpenGLError(__FILE__, __LINE__);




    //Create buffer data..
    GLsizei    stride = 0;
    GLintptr   memoryOffset=0;
    GLsizeiptr totalBufferDataSize=sizeOfVertices+sizeOfTextureCoords+sizeOfColors+sizeOfNormals;
    //----------------------------------------------------------------------------------------------------------------------------
    glBufferData(GL_ARRAY_BUFFER,totalBufferDataSize,NULL,GL_STATIC_DRAW);                   checkOpenGLError(__FILE__, __LINE__);
    //----------------------------------------------------------------------------------------------------------------------------
    if ((vertices!=0) && (sizeOfVertices!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfVertices, vertices);               checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfVertices;
     //stride += 3 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (textureCoords!=0) && (sizeOfTextureCoords!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, memoryOffset, sizeOfTextureCoords , textureCoords );  checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfTextureCoords;
     //stride += 2 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (colors!=0) && (sizeOfColors!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfColors, colors);                   checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfColors;
     //stride += 3 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if ( (normals!=0) && (sizeOfNormals!=0) )
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfNormals, normals);                 checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfNormals;
     //stride += 3 * sizeof(float);
    }
    //----------------------------------------------------------------------------------------------------------------------------


    memoryOffset=0;

    //----------------------------------------------------------------------------------------------------------------------------
    if ((vertices!=0) && (sizeOfVertices!=0))
    {
     //Pass vPosition to shader
     GLuint vPosition = glGetAttribLocation(programID, "vPosition" );                        checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vPosition) && (vPosition!=-1) )
     {
      glEnableVertexAttribArray(vPosition);                                                  checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vPosition,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);    checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfVertices;
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
      glVertexAttribPointer(vTexture,2,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);     checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfTextureCoords;
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
      glVertexAttribPointer(vColor,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);       checkOpenGLError(__FILE__, __LINE__);
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
      glVertexAttribPointer(vNormal,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);      checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=sizeOfNormals;
     }
    }
    //----------------------------------------------------------------------------------------------------------------------------

   return 1;
  #else
   #warning "USE_GLEW not defined, pushObjectToBufferData will not be included.."
   return 0;
  #endif
}
