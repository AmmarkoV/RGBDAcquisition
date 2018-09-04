#ifndef COMPUTESHADER_H_INCLUDED
#define COMPUTESHADER_H_INCLUDED


/**
* @brief The structure that defines what a Compute Shader Consists of
*/
struct computeShaderObject
{
//  GLuint computeShader;
  int computeShaderObject;
  int computeShaderProgram;

  char * compMem;
  unsigned long compMemLength;
};


int unloadComputeShader(struct computeShaderObject * sh);
struct computeShaderObject  * loadComputeShader(char * computeShaderPath);

#endif // COMPUTESHADER_H_INCLUDED
