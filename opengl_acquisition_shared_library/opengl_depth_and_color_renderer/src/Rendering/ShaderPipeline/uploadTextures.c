#include <stdio.h>

#if USE_GLEW
// Include GLEW
#include <GL/glew.h>
#endif // USE_GLEW

#include "uploadTextures.h"
#include "../../Tools/tools.h"

//#include <GL/gl.h>  //Also on header..
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>



int uploadColorImageAsTexture(
                               GLuint programID,
                               GLuint *textureID,
                               unsigned int * alreadyUploaded,
                               unsigned char * colorPixels,
                               unsigned int colorWidth ,
                               unsigned int colorHeight ,
                               unsigned int colorChannels ,
                               unsigned int colorBitsperpixel
                              )
{
  glUseProgram(programID);

    if (*alreadyUploaded)
     {
       glDeleteTextures(1,textureID);
       *alreadyUploaded=0;
     }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,textureID);
    *alreadyUploaded=1;
    glBindTexture(GL_TEXTURE_2D,*textureID);

      /* LOADING TEXTURE --WITHOUT-- MIPMAPING - IT IS LOADED RAW*/
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);                       //GL_RGB
      checkOpenGLError(__FILE__, __LINE__);
      glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    colorWidth ,
                    colorHeight,
                    0,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    (const GLvoid *) colorPixels
                  );
      checkOpenGLError(__FILE__, __LINE__);

    glFlush();
    return 1;
}


int uploadDepthImageAsTexture(
                               GLuint programID,
                               GLuint *textureID,
                               unsigned int * alreadyUploaded,
                               unsigned short * depthPixels,
                               unsigned int depthWidth,
                               unsigned int depthHeight,
                               unsigned int depthChannels,
                               unsigned int depthBitsPerPixel
                              )
{
  glUseProgram(programID);

    if (*alreadyUploaded)
     {
       glDeleteTextures(1,textureID);
       *alreadyUploaded=0;
     }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,textureID);
    *alreadyUploaded=1;
    glBindTexture(GL_TEXTURE_2D,*textureID);

      /* LOADING TEXTURE --WITHOUT-- MIPMAPING - IT IS LOADED RAW*/
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);                       //GL_RGB
      checkOpenGLError(__FILE__, __LINE__);
      glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_R16UI,
                    depthWidth ,
                    depthHeight,
                    0,
                    GL_RED_INTEGER,
                    GL_UNSIGNED_SHORT,
                    (const GLvoid *) depthPixels
                  );
      checkOpenGLError(__FILE__, __LINE__);

    glFlush();
    return 1;
}

