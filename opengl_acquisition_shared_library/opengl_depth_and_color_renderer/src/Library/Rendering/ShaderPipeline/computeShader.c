#include "computeShader.h"


#include <stdio.h>
#include <stdlib.h>

#if USE_GLEW
// Include GLEW
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#else
 #warning "USE_GLEW not defined , shader code is useless.."
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#endif // USE_GLEW





int unloadComputeShader(struct computeShaderObject * sh)
{
   fprintf(stderr,"unloadComputeShader \n");
   glUseProgram(0);

    glDetachShader(sh->computeShaderProgram, sh->computeShaderObject);

    glDeleteShader(sh->computeShaderObject);
    //glDeleteShader(sh->computeShader);
    glDeleteProgram(sh->computeShaderProgram);

   if (sh->compMem!=0) { free(sh->compMem); }
   if (sh!=0) { free(sh); }
   return 1;
}


struct computeShaderObject  * loadComputeShader(char * computeShaderPath)
{
  fprintf(stderr,"Loading Compute Shaders ( %s ) \n",computeShaderPath);

  GLint isok;
  struct computeShaderObject  * sh = (struct computeShaderObject *)  malloc(sizeof(struct computeShaderObject));


  sh->computeShaderObject = glCreateShader(GL_COMPUTE_SHADER);

  sh->compMem=loadShaderFileToMem(computeShaderPath,&sh->compMemLength);
  if ( (sh->compMem==0)||(sh->compMemLength==0)) { fprintf(stderr,"Could not load compute shader in memory..\n"); }

  fprintf(stderr,"COMPUTE SHADER (%lu bytes long) \n\n%s\n",sh->compMemLength,sh->compMem);
  glShaderSource(sh->computeShaderObject, 1, &sh->compMem, &sh->compMemLength);
  glCompileShader(sh->computeShaderObject);
  glGetShaderiv(sh->computeShaderObject, GL_COMPILE_STATUS, &isok);
  if (!isok)
    {
      fprintf(stderr,"Could not compile  compute shader %s \n",computeShaderPath);

      GLchar info[1024]; GLsizei length;
      glGetShaderInfoLog(sh->computeShaderObject,1024, &length,&info);
      fprintf(stderr,"Shader error : %s \n",info);

      free(sh->compMem); free(sh);
      return 0;
    }



   sh->computeShaderProgram = glCreateProgram();

   glAttachShader(sh->computeShaderProgram, sh->computeShaderObject);

   glLinkProgram(sh->computeShaderProgram);

   glGetProgramiv(sh->computeShaderProgram, GL_LINK_STATUS, &isok);
   if (!isok)
   {
     fprintf(stderr,"Could not link shaders\n");

      GLchar info[1024]; GLsizei length;
      glGetProgramInfoLog(sh->computeShaderProgram, 1024  , &length , &info);
      fprintf(stderr,"Shader error : %s \n",info);

      unloadComputeShader(sh);
      return 0;
   }

   return sh;
}
