#include <stdio.h>
#include <stdlib.h>

#include <GL/glx.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include "shader_loader.h"
#include "../../Tools/tools.h"


#define USE_FAILSAFE_SHADER 0

static const char *vertex_source =
{
"void main(){"
"gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;"
"texture_coordinate = vec2(gl_MultiTexCoord0);"
"}"
};

// a simple fragment shader source
// this change the fragment's color by yellow color
static const char *fragment_source =
{
"void main(void){"
"   gl_FragColor = texture2D(my_color_texture, texture_coordinate);"
"}"
};


int useShader(struct shaderObject * shader)
{
  return glUseProgramObjectARB(shader->ProgramObject);
}

struct shaderObject * loadShader(char * vertexShaderChar,char * fragmentShaderChar)
{
  fprintf(stderr,"Loading Shaders ( %s , %s ) \n",vertexShaderChar,fragmentShaderChar);

  GLint isok;
  struct shaderObject * sh = (struct shaderObject *)  malloc(sizeof(struct shaderObject));


  sh->ProgramObject = glCreateProgram();

  sh->vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
  sh->fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

  sh->vertMem=loadFileToMem(vertexShaderChar,&sh->vertMemLength);
  if ( (sh->vertMem==0)||(sh->vertMemLength==0)) { fprintf(stderr,"Could not load vertex shader in memory..\n"); }

  sh->fragMem=loadFileToMem(fragmentShaderChar,&sh->fragMemLength);
  if ( (sh->fragMem==0)||(sh->fragMemLength==0)) { fprintf(stderr,"Could not load fragment shader in memory..\n"); }

  fprintf(stderr,"VERTEX SHADER (%lu bytes long) \n\n %s \n",sh->vertMemLength,sh->vertMem);
  fprintf(stderr,"FRAGMENT SHADER (%lu bytes long) \n\n %s \n",sh->fragMemLength,sh->fragMem);


  #if USE_FAILSAFE_SHADER==1
   fprintf(stderr,"Overriding %s and %s and using failsafe shaders instead!\n",vertexShaderChar,fragmentShaderChar);
   glShaderSource(sh->vertexShaderObject, 1, &vertex_source , 0 );
   glShaderSource(sh->fragmentShaderObject, 1, &fragment_source , 0 );
  #else
   glShaderSource(sh->vertexShaderObject, 1,&sh->vertMem, &sh->vertMemLength);
   glShaderSource(sh->fragmentShaderObject, 1, &sh->fragMem, &sh->fragMemLength);
  #endif

  glCompileShader(sh->vertexShaderObject);
  glGetShaderiv(sh->vertexShaderObject, GL_COMPILE_STATUS, &isok);
  if (!isok)
    {
      fprintf(stderr,"Could not compile shader %s \n",vertexShaderChar);

      GLchar info[1024]; GLsizei length;
      glGetShaderInfoLog(sh->vertexShaderObject,1024, &length,&info);
      fprintf(stderr,"Shader error : %s \n",info);

      glDeleteProgram(sh->ProgramObject);
      free(sh->vertMem); free(sh->fragMem); free(sh);
      return 0;
    }

  glCompileShader(sh->fragmentShaderObject);
  glGetShaderiv(sh->fragmentShaderObject, GL_COMPILE_STATUS, &isok);
  if (!isok)
    {
      fprintf(stderr,"Could not compile shader %s \n",fragmentShaderChar);

      GLchar info[1024]; GLsizei length;
      glGetShaderInfoLog(sh->fragmentShaderObject,1024, &length,&info);
      fprintf(stderr,"Shader error : %s \n",info);

      glDeleteShader(sh->vertexShaderObject); //(because since we are here it means that vertexShader was ok)  :(
      glDeleteProgram(sh->ProgramObject);
      free(sh->vertMem); free(sh->fragMem); free(sh);
      return 0;
    }


   glAttachShader(sh->ProgramObject, sh->vertexShaderObject);
   glAttachShader(sh->ProgramObject, sh->fragmentShaderObject);

   glLinkProgram(sh->ProgramObject);

   glGetProgramiv(sh->ProgramObject, GL_LINK_STATUS, &isok);
   if (!isok)
   {
     fprintf(stderr,"Could not link shaders\n");

      GLchar info[1024]; GLsizei length;
      glGetProgramInfoLog(sh->ProgramObject, 1024  , &length , &info);
      fprintf(stderr,"Shader error : %s \n",info);

      glDetachShader(sh->ProgramObject, sh->vertexShaderObject);
      glDetachShader(sh->ProgramObject, sh->fragmentShaderObject);
      glDeleteShader(sh->vertexShaderObject);
      glDeleteShader(sh->fragmentShaderObject);
      glDeleteProgram(sh->ProgramObject);

      free(sh->vertMem); free(sh->fragMem); free(sh);
      return 0;
   }


   fprintf(stderr,"Switching to new shaders \n");
   glUseProgramObjectARB(sh->ProgramObject);


   return sh;
}

int unloadShader(struct shaderObject * sh)
{
   glUseProgram(0);

    glDetachShader(sh->ProgramObject, sh->vertexShaderObject);
    glDetachShader(sh->ProgramObject, sh->fragmentShaderObject);

    glDeleteShader(sh->vertexShaderObject);
    glDeleteShader(sh->fragmentShaderObject);
    glDeleteProgram(sh->ProgramObject);

   if (sh->vertMem!=0) { free(sh->vertMem); }
   if (sh->fragMem!=0) { free(sh->fragMem); }
   if (sh!=0) { free(sh); }
   return 1;
}



