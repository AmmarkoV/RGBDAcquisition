#ifndef SHADER_LOADER_H_INCLUDED
#define SHADER_LOADER_H_INCLUDED


#include <GL/glx.h>
#include <GL/gl.h>
#include <GL/glext.h>

struct shaderObject
{
  GLuint vertexShader, fragmentShader;
  int ProgramObject;
  int vertexShaderObject;
  int fragmentShaderObject;

  char * vertMem;
  unsigned long vertMemLength;

  char * fragMem;
  unsigned long fragMemLength;
};



struct shaderObject * loadShader(char * vertexShaderChar,char * fragmentShaderChar);
int unloadShader(struct shaderObject * sh);

#endif // SHADER_LOADER_H_INCLUDED
