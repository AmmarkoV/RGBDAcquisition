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
    //fprintf(stderr,"PushBones(triangle count %u,Number Of Bones %u,Bones Index Size %u)\n",shaderData->triangleCount,shaderData->numberOfBones,shaderData->sizeOfBoneIndexes);
    //fprintf(stderr,"Normal Size %u,Color Size %u, Vertex Size %u\n",shaderData->sizeOfNormals,shaderData->sizeOfColors,shaderData->sizeOfVertices);

    if (shaderData->numberOfBonesPerVertex>4)
    {
       fprintf(stderr,"Too many bones per vertex (%u) ",shaderData->numberOfBonesPerVertex);
       return 0;
    }

    //If no data given, zero their size..
    if (shaderData->vertices==0)         { shaderData->sizeOfVertices=0;         }
    if (shaderData->normals==0)          { shaderData->sizeOfNormals=0;          }
    if (shaderData->textureCoords==0)    { shaderData->sizeOfTextureCoords=0;    }
    if (shaderData->colors==0)           { shaderData->sizeOfColors=0;           }
    if (shaderData->indices==0)          { shaderData->sizeOfIndices=0;          }
    if (shaderData->boneIndexes==0)      { shaderData->sizeOfBoneIndexes=0;      }
    if (shaderData->boneWeightValues==0) { shaderData->sizeOfBoneWeightValues=0; }
    if (shaderData->boneTransforms==0)   { shaderData->sizeOfBoneTransforms=0;   }

    //Select Shader to render with
    glUseProgram(programID);                       checkOpenGLError(__FILE__, __LINE__);



    //First of all pass the uniform Bone Transforms!
    //if  (shaderData->timestamp < shaderData->lastTimestampBoneModification)
    {
      //Sanity check on our OpenGL context to make sure we are in spec
      GLint maxUniformMatricesOnGLSLShader=0;
      glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUniformMatricesOnGLSLShader);
      maxUniformMatricesOnGLSLShader = maxUniformMatricesOnGLSLShader / (16*sizeof(float));
      if (maxUniformMatricesOnGLSLShader<=shaderData->numberOfBones)
      {
        fprintf(stderr, "Your Geometry exceeds the maximum number of uniform matrices which is %u mat4\n",maxUniformMatricesOnGLSLShader );
      }


      shaderData->lastTimestampBoneModification = shaderData->timestamp + 1;
      if  (shaderData->sizeOfBoneTransforms!=0)
      {
       //Pass vBoneTransform to shader
       GLuint vBoneTransform = glGetUniformLocation(programID, "vBoneTransform");                        checkOpenGLError(__FILE__, __LINE__);
       if ( (GL_INVALID_OPERATION != vBoneTransform) && (vBoneTransform!=-1) )
        {
         glUniformMatrix4fv(
                             vBoneTransform,
                             shaderData->numberOfBones,
                             GL_TRUE, //We NEED Transpose since our matrices are not in an OGL major format
                             shaderData->boneTransforms
                           );
         checkOpenGLError(__FILE__, __LINE__);
        }
       }
    }

    //Take care of VAO, ElementBuffer and ArrayBuffer
    //------------------------------------------------------------------------------------------------------------------------------------------------
    if (generateNewVao)
      { glGenVertexArrays(1,&shaderData->VAO);     checkOpenGLError(__FILE__, __LINE__); }
    glBindVertexArray(shaderData->VAO);            checkOpenGLError(__FILE__, __LINE__);
    //------------------------------------------------------------------------------------------------------------------------------------------------
    if (generateNewElementBuffer)
      { glGenBuffers(1, &shaderData->elementBuffer);            checkOpenGLError(__FILE__, __LINE__); }

    if (shaderData->sizeOfIndices!=0)
    {
     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shaderData->elementBuffer);                                       checkOpenGLError(__FILE__, __LINE__);
     glBufferData(GL_ELEMENT_ARRAY_BUFFER, shaderData->sizeOfIndices, shaderData->indices , GL_STATIC_DRAW); checkOpenGLError(__FILE__, __LINE__);
    }
    //------------------------------------------------------------------------------------------------------------------------------------------------
    // Create and initialize a buffer object on the server side (GPU)
    if (generateNewArrayBuffer)
     { glGenBuffers( 1, &shaderData->arrayBuffer );                   checkOpenGLError(__FILE__, __LINE__); }
    glBindBuffer( GL_ARRAY_BUFFER, shaderData->arrayBuffer );         checkOpenGLError(__FILE__, __LINE__);
    //------------------------------------------------------------------------------------------------------------------------------------------------


    //if  (shaderData->timestamp < shaderData->lastTimestampBaseModification)
    if (generateNewArrayBuffer)
    {
      fprintf(stderr,"PushBones(triangle count %u,Number Of Bones %u,Bones Index Size %u)\n",shaderData->triangleCount,shaderData->numberOfBones,shaderData->sizeOfBoneIndexes);
      shaderData->lastTimestampBaseModification = shaderData->timestamp + 1;
      //Create buffer holder data..
      GLsizei    stride = 0;
      GLintptr   memoryOffset=0;
      GLsizeiptr totalBufferDataSize=shaderData->sizeOfVertices+
                                     shaderData->sizeOfTextureCoords+
                                     shaderData->sizeOfColors+
                                     shaderData->sizeOfNormals+
                                     shaderData->sizeOfBoneIndexes+
                                     shaderData->sizeOfBoneWeightValues;
     //----------------------------------------------------------------------------------------------------------------------------
     glBufferData(GL_ARRAY_BUFFER,totalBufferDataSize,NULL,GL_STATIC_DRAW);                                           checkOpenGLError(__FILE__, __LINE__);
     //----------------------------------------------------------------------------------------------------------------------------


     //Create sub buffers inside buffer data..
     //----------------------------------------------------------------------------------------------------------------------------
     //----------------------------------------------------------------------------------------------------------------------------
     if (shaderData->sizeOfVertices!=0)
     {
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfVertices, shaderData->vertices);               checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfVertices;
     }
     //----------------------------------------------------------------------------------------------------------------------------
     if (shaderData->sizeOfTextureCoords!=0)
     {
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfTextureCoords , shaderData->textureCoords);    checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfTextureCoords;
     }
     //----------------------------------------------------------------------------------------------------------------------------
     if (shaderData->sizeOfColors!=0)
     {
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfColors, shaderData->colors);                   checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfColors;
     }
     //----------------------------------------------------------------------------------------------------------------------------
     if (shaderData->sizeOfNormals!=0)
     {
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfNormals, shaderData->normals);                 checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfNormals;
     }
     //----------------------------------------------------------------------------------------------------------------------------
     //----------------------------------------------------------------------------------------------------------------------------
     //   now upload bones..
     //----------------------------------------------------------------------------------------------------------------------------
     //----------------------------------------------------------------------------------------------------------------------------
     if  (shaderData->sizeOfBoneIndexes!=0)
     {//GL_ELEMENT_ARRAY_BUFFER
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfBoneIndexes, shaderData->boneIndexes);                         checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfBoneIndexes;
     }
     //----------------------------------------------------------------------------------------------------------------------------
     if (shaderData->sizeOfBoneWeightValues!=0)
     {
      glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, shaderData->sizeOfBoneWeightValues, shaderData->boneWeightValues);              checkOpenGLError(__FILE__, __LINE__);
      memoryOffset+=shaderData->sizeOfBoneWeightValues;
     }
     //----------------------------------------------------------------------------------------------------------------------------


     //Restart count
     memoryOffset=0;




    if (shaderData->sizeOfVertices!=0)
    {
     //Pass vPosition to shader
     GLuint vPosition = glGetAttribLocation(programID, "vPosition" );                        checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vPosition) && (vPosition!=-1) )
     {
      glEnableVertexAttribArray(vPosition);                                                  checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vPosition,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);    checkOpenGLError(__FILE__, __LINE__);
     }
      memoryOffset+=shaderData->sizeOfVertices;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfTextureCoords!=0)
    {
     //Pass vTexture to shader
     GLuint vTexture = glGetAttribLocation(programID, "vTexture");                           checkOpenGLError(__FILE__, __LINE__);
    //GLuint textureStrengthLocation = glGetUniformLocation(programID, "textureStrength");  checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vTexture ) && (vTexture!=-1) )
     {
      glEnableVertexAttribArray(vTexture);                                                   checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vTexture,2,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);     checkOpenGLError(__FILE__, __LINE__);
     }
      memoryOffset+=shaderData->sizeOfTextureCoords;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfColors!=0)
    {
     //Pass vColor to shader
     GLuint vColor = glGetAttribLocation(programID, "vColor");                               checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vColor) && (vColor!=-1) )
     {
      glEnableVertexAttribArray(vColor);                                                     checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vColor,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);       checkOpenGLError(__FILE__, __LINE__);
     }
     memoryOffset+=shaderData->sizeOfColors;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfNormals!=0)
    {
     //Pass vNormal to shader
     GLuint vNormal = glGetAttribLocation(programID, "vNormal");                             checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION != vNormal) && (vNormal!=-1) )
     {
      glEnableVertexAttribArray(vNormal);                                                    checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vNormal,3,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);      checkOpenGLError(__FILE__, __LINE__);
     }
     memoryOffset+=shaderData->sizeOfNormals;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (shaderData->sizeOfBoneIndexes!=0)
    {
     //Pass vBoneIndexIDs to shader
     GLuint vBoneIndexIDs = glGetAttribLocation(programID, "vBoneIndexIDs" );                                                         checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vBoneIndexIDs) && (vBoneIndexIDs!=-1) )
     {
      glEnableVertexAttribArray(vBoneIndexIDs);                                                                                       checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribIPointer(vBoneIndexIDs,shaderData->numberOfBonesPerVertex,GL_UNSIGNED_INT,stride,(GLvoid*) memoryOffset);         checkOpenGLError(__FILE__, __LINE__);
     }
     memoryOffset+=shaderData->sizeOfBoneIndexes;
    }
    //----------------------------------------------------------------------------------------------------------------------------
   if (shaderData->sizeOfBoneWeightValues!=0)
    {
     //Pass vBoneWeightValues to shader
     GLuint vBoneWeightValues = glGetAttribLocation(programID, "vBoneWeightValues");                                                  checkOpenGLError(__FILE__, __LINE__);
     if ( (GL_INVALID_OPERATION!=vBoneWeightValues) && (vBoneWeightValues!=-1) )
     {
      glEnableVertexAttribArray(vBoneWeightValues);                                                                                   checkOpenGLError(__FILE__, __LINE__);
      glVertexAttribPointer(vBoneWeightValues,shaderData->numberOfBonesPerVertex,GL_FLOAT,GL_FALSE,stride,(GLvoid*) memoryOffset);    checkOpenGLError(__FILE__, __LINE__);
     }
     memoryOffset+=shaderData->sizeOfBoneWeightValues;
    }
    //----------------------------------------------------------------------------------------------------------------------------
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
    if (sizeOfVertices!=0)
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfVertices, vertices);               checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfVertices;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (sizeOfTextureCoords!=0)
    {
     glBufferSubData( GL_ARRAY_BUFFER, memoryOffset, sizeOfTextureCoords , textureCoords );  checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfTextureCoords;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (sizeOfColors!=0)
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfColors, colors);                   checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfColors;
    }
    //----------------------------------------------------------------------------------------------------------------------------
    if (sizeOfNormals!=0)
    {
     glBufferSubData(GL_ARRAY_BUFFER, memoryOffset, sizeOfNormals, normals);                 checkOpenGLError(__FILE__, __LINE__);
     memoryOffset+=sizeOfNormals;
    }
    //----------------------------------------------------------------------------------------------------------------------------

    memoryOffset=0;

    //----------------------------------------------------------------------------------------------------------------------------
    if (sizeOfVertices!=0)
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
    if (sizeOfTextureCoords!=0)
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
    if (sizeOfColors!=0)
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
    if (sizeOfNormals!=0)
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
