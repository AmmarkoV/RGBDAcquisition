
#if USE_GLEW
// Include GLEW
#include <GL/glew.h>
#else
 #warning "Please use the -DUSE_GLEW define for the GL shader code.."
#endif // USE_GLEW

#include "render_buffer.h"

#include <stdio.h>

//#include <GL/gl.h>  //Also on header..
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include "../../Tools/tools.h"  //for GetTickCountMilliseconds();
#include "../../../../../../tools/AmMatrix/matrixOpenGL.h" //for getModelViewProjectionMatrixFromMatrices();



int checkIfFrameBufferIsOk(const char * label, GLuint framebufferName)
{
    #if USE_GLEW
    GLenum res;

    res = glCheckFramebufferStatus(GL_FRAMEBUFFER );
    //res = glCheckNamedFramebufferStatus(GL_FRAMEBUFFER,framebufferName);
    if (res==0) { return 1;}
    if (res==GL_FRAMEBUFFER_COMPLETE) { return 1; }


    fprintf(stderr,"%s\n",label);

    switch (res)
    {
      case GL_FRAMEBUFFER_UNDEFINED :
        fprintf(stderr," The specified framebuffer is the default read or draw framebuffer, but the default framebuffer does not exist.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT :
        fprintf(stderr," Any of the framebuffer attachment points are framebuffer incomplete.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT :
        fprintf(stderr," The framebuffer does not have at least one image attached to it.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER :
        fprintf(stderr," The value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAW_BUFFERi.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER :
        fprintf(stderr," GL_READ_BUFFER is not GL_NONE and the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.\n");
      break;

      case GL_FRAMEBUFFER_UNSUPPORTED :
        fprintf(stderr," The combination of internal formats of the attached images violates an implementation-dependent set of restrictions.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE :
        fprintf(stderr," The value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES \
        is the not same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES.\n");

        fprintf(stderr," is also returned if the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached textures;\
         or, if the attached images are a mix of renderbuffers and textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all attached textures.\n");
      break;

      case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS :
        fprintf(stderr," Any framebuffer attachment is layered, and any populated attachment is not layered, or if all populated color attachments are not from textures of the same target.\n");
      break;

      case GL_FRAMEBUFFER_COMPLETE :
         return 1;
      break;

      default :
        fprintf(stderr," Unknown error ( %u )\n",res);
      break;
    };

    return 0;
  #else
   return 1;
  #endif
}




int initializeFramebuffer(
                          GLuint * FramebufferName,
                          GLuint * renderedTexture,
                          GLuint * depthTexture,
                          unsigned int width,
                          unsigned int height
                         )
{
    #if USE_GLEW
    // ---------------------------------------------
    // Render to Texture - specific code begins here
    // ---------------------------------------------
    //fprintf(stderr," initializeFramebuffer running.. \n");

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    glGenFramebuffers(1, FramebufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, *FramebufferName);


    //fprintf(stderr," glGenTextures.. \n");
    // The texture we're going to render to
    glGenTextures(1, renderedTexture);

    //fprintf(stderr," glBindTexture.. \n");
    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, *renderedTexture);

    //fprintf(stderr," glTexImage2D.. \n");
    // Give an empty image to OpenGL ( the last "0" means "empty" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);

    // Poor filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //fprintf(stderr," glFramebufferTexture.. \n");
    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, *renderedTexture, 0);

    //// Alternative : Depth texture. Slower, but you can sample it later in your shader
    if (depthTexture!=0)
    {
     // The depth buffer
     GLuint depthrenderbuffer;
     glGenRenderbuffers(1, &depthrenderbuffer);
     glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
     glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
     glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

     glGenTextures(1, depthTexture);
     glBindTexture(GL_TEXTURE_2D, *depthTexture);
     glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT24, width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0); //GL_UNSIGNED_BYTE
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

     //// Depth texture alternative :
     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D, *depthTexture, 0);
    }


    //fprintf(stderr," ready to pass draw buffer..\n");


    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    return checkIfFrameBufferIsOk(" initializeFramebuffer : Checking framebuffer failed..",*FramebufferName);

  #else
   return 0;
  #endif
}




int drawFramebufferFromTexture(
                               GLuint FramebufferName,
                               GLuint textureToDraw,
                               GLuint programFrameBufferID,
                               GLuint quad_vertexbuffer,
                               //-----------------------
                               GLuint texID,
                               GLuint timeID,
                               GLuint resolutionID,
                               unsigned int width,
                               unsigned int height
                              )
{
    #if USE_GLEW
        // Render to the screen
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
        glViewport(0,0,width,height); // Render on the whole framebuffer, complete from the lower left corner to the upper right

        glClearColor( 0, 0.0, 0, 1 );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);         // Clear the screen

        checkIfFrameBufferIsOk("drawFramebufferFromTexture framebuffer error:",FramebufferName);
        // Render on the whole framebuffer, complete from the lower left corner to the upper right

        // Clear the screen

        // Use our shader
        glUseProgram(programFrameBufferID);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureToDraw);
        // Set our "renderedTexture" sampler to use Texture Unit 0
        glUniform1i(texID, 0);



        glUniform1f(timeID, (float)(GetTickCountMilliseconds()/1000.0f) );
        glUniform3f(resolutionID, width, height , 1);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
        glVertexAttribPointer(
                              0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                              3,                  // size
                              GL_FLOAT,           // type
                              GL_FALSE,           // normalized?
                              0,                  // stride
                              (void*) 0           // array buffer offset
                             );

        // Draw the triangles !
        glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles

        glDisableVertexAttribArray(0);
  return 1;
  #else
   return 0;
  #endif
}



int drawFramebufferTexToTex(
                            GLuint FramebufferName,
                            GLuint programFrameBufferID,
                            GLuint quad_vertexbuffer,
                            GLuint renderedTexture,
                            GLuint texID,
                            GLuint timeID,
                            GLuint resolutionID,
                            unsigned int width,
                            unsigned int height
                           )
{
 return drawFramebufferFromTexture(
                                   FramebufferName,
                                   renderedTexture,
                                   programFrameBufferID,
                                   quad_vertexbuffer,
                                   texID,
                                   timeID,
                                   resolutionID,
                                   width,
                                   height
                                  );
}

int drawFramebufferToScreen(
                            GLuint programFrameBufferID,
                            GLuint quad_vertexbuffer,
                            GLuint renderedTexture,
                            GLuint texID,
                            GLuint timeID,
                            GLuint resolutionID,
                            unsigned int width,
                            unsigned int height
                           )
{
 return drawFramebufferFromTexture(
                                    0,//We want to render TO SCREEN
                                    renderedTexture,
                                    programFrameBufferID,
                                    quad_vertexbuffer,
                                    texID,
                                    timeID,
                                    resolutionID,
                                    width,
                                    height
                                  );
}




int drawVertexArrayWithMVPMatrices(
                                   GLuint programID,
                                   GLuint vao,
                                   GLuint MatrixID,
                                   GLuint TextureID,
                                   unsigned int triangleCount,
                                   unsigned int elementCount,
                                   //-----------------------------------------
                                   struct Matrix4x4OfFloats * modelMatrix,
                                   //-----------------------------------------
                                   struct Matrix4x4OfFloats * projectionMatrix,
                                   struct Matrix4x4OfFloats * viewportMatrix,
                                   struct Matrix4x4OfFloats * viewMatrix,
                                   //-----------------------------------------
                                   char renderWireframe
                                  )
{
  //Select Shader to render with
  glUseProgram(programID);                  checkOpenGLError(__FILE__, __LINE__);

  //Select Vertex Array Object To Render
  glBindVertexArray(vao);                   checkOpenGLError(__FILE__, __LINE__);

  //-------------------------------------------------------------------
  struct Matrix4x4OfFloats MVP;
  getModelViewProjectionMatrixFromMatrices(&MVP,projectionMatrix,viewMatrix,modelMatrix);
  transpose4x4FMatrix(MVP.m); //OpenGL needs a column-major/row-major flip..
  //-------------------------------------------------------------------

  glPushAttrib(GL_ALL_ATTRIB_BITS);
  //Our flipped view needs front culling..
  glCullFace(GL_FRONT);
  glEnable(GL_CULL_FACE);

  //-------------------------------------------------
  if (renderWireframe) { glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); checkOpenGLError(__FILE__, __LINE__); } else
                       { glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); checkOpenGLError(__FILE__, __LINE__); }
  //-------------------------------------------------

  GLint textureLocation = 0;
  if (TextureID!=0)
    {
      GLint useTexture = glGetUniformLocation(programID, "useTexture");     checkOpenGLError(__FILE__, __LINE__);
      glUniform1f(useTexture, 1.0);                                         checkOpenGLError(__FILE__, __LINE__);

      //fprintf(stderr,"renderingTexture %u\n",TextureID);
      glEnable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE0);                                         checkOpenGLError(__FILE__, __LINE__);
      glBindTexture(GL_TEXTURE_2D,TextureID);                               checkOpenGLError(__FILE__, __LINE__);
      textureLocation = glGetUniformLocation(programID, "renderedTexture"); checkOpenGLError(__FILE__, __LINE__);
      glUniform1i(textureLocation, 0);                                      checkOpenGLError(__FILE__, __LINE__);
    }

  // Send our transformation to the currently bound shader, in the "MVP" uniform
  glUniformMatrix4fv(MatrixID, 1, GL_FALSE/*TRANSPOSE*/,MVP.m);



  if (elementCount!=0)
  {
    glDrawElements(GL_TRIANGLES,  elementCount ,GL_UNSIGNED_INT,(void*)0);   checkOpenGLError(__FILE__, __LINE__);
  } else
  {
    glDrawArrays(GL_TRIANGLES, 0, triangleCount );                           checkOpenGLError(__FILE__, __LINE__);
  }

  if (TextureID!=0)
    {
      glActiveTexture(GL_TEXTURE0);    checkOpenGLError(__FILE__, __LINE__);
      glBindTexture(GL_TEXTURE_2D,0);  checkOpenGLError(__FILE__, __LINE__);
      glUniform1i(textureLocation, 0); checkOpenGLError(__FILE__, __LINE__);
    }

  glPopAttrib();

  glBindVertexArray(0); checkOpenGLError(__FILE__, __LINE__);
  return 1;
}






