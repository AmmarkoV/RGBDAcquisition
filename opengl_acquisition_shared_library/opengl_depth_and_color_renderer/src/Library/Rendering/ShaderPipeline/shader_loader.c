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


char * loadShaderFileToMem(char * filename,unsigned long * file_length)
{
  if (filename==0)  { fprintf(stderr,"Could not load shader incorrect filename \n"); return 0; }
  if (file_length==0)  { fprintf(stderr,"Could not load shader %s , incorrect file length parameter \n",filename); return 0; }

  FILE * pFile;
  long lSize;
  char * buffer;
  size_t result;

  pFile = fopen ( filename , "rb" );
  if (pFile==0) { fprintf(stderr,"Could not open shader file %s \n",filename); return 0;}

  // obtain file size :
  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  // allocate memory to contain the whole file:
  buffer = (char*) malloc ( (sizeof(char)*lSize)+1 );
  if (buffer == 0) {fprintf(stderr,"Could not allocate %u bytes of memory for shader file %s \n",(unsigned int ) lSize,filename);   fclose (pFile); return 0; }

  // copy the file into the buffer:
  result = fread (buffer,1,lSize,pFile);
  if (result != lSize) {fputs ("Reading error",stderr); free(buffer);  fclose (pFile); return 0; }

  /* the whole file is now loaded in the memory buffer. */

  // terminate
  fclose (pFile);

  buffer[lSize]=0; //Add a null termination for shader usage
  *file_length = lSize;
  return buffer;
}


int useShader(struct shaderObject * shader)
{
 // return glUseProgramObjectARB(shader->ProgramObject);
 return 0;
}

struct shaderObject * loadShader(char * vertexShaderChar,char * fragmentShaderChar)
{
  fprintf(stderr,"Loading Shaders ( %s , %s ) \n",vertexShaderChar,fragmentShaderChar);

  GLint isok;
  struct shaderObject * sh = (struct shaderObject *)  malloc(sizeof(struct shaderObject));


  sh->ProgramObject = glCreateProgram();

  sh->vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
  sh->fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

  sh->vertMem=loadShaderFileToMem(vertexShaderChar,&sh->vertMemLength);
  if ( (sh->vertMem==0)||(sh->vertMemLength==0)) { fprintf(stderr,"Could not load vertex shader in memory..\n"); }

  sh->fragMem=loadShaderFileToMem(fragmentShaderChar,&sh->fragMemLength);
  if ( (sh->fragMem==0)||(sh->fragMemLength==0)) { fprintf(stderr,"Could not load fragment shader in memory..\n"); }

  //fprintf(stderr,"VERTEX SHADER (%lu bytes long) \n\n%s\n",sh->vertMemLength,sh->vertMem);
  //fprintf(stderr,"FRAGMENT SHADER (%lu bytes long) \n\n%s\n",sh->fragMemLength,sh->fragMem);


  #if USE_FAILSAFE_SHADER==1
   fprintf(stderr,"Overriding %s and %s and using failsafe shaders instead!\n",vertexShaderChar,fragmentShaderChar);
   glShaderSource(sh->vertexShaderObject, 1,(const GLchar **) &vertex_source , 0 );
   glShaderSource(sh->fragmentShaderObject, 1,(const GLchar **) &fragment_source , 0 );
  #else
   GLint vertMemLength = (GLint) sh->vertMemLength; //We do this cast to make sure arguments signatures match..
   glShaderSource(sh->vertexShaderObject, 1,(const GLchar **) &sh->vertMem, &vertMemLength);

   GLint fragMemLength = (GLint) sh->fragMemLength; //We do this cast to make sure arguments signatures match..
   glShaderSource(sh->fragmentShaderObject, 1,(const GLchar **) &sh->fragMem, &fragMemLength);
  #endif

  glCompileShader(sh->vertexShaderObject);
  glGetShaderiv(sh->vertexShaderObject, GL_COMPILE_STATUS, &isok);
  if (!isok)
    {
      fprintf(stderr,"Could not compile shader %s \n",vertexShaderChar);

      GLchar info[1024]; GLsizei length;
      glGetShaderInfoLog(sh->vertexShaderObject,1024, &length,info);
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
      glGetShaderInfoLog(sh->fragmentShaderObject,1024, &length,info);
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
      glGetProgramInfoLog(sh->ProgramObject, 1024  , &length , info);
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



