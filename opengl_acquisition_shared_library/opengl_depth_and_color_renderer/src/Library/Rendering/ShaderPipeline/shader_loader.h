/** @file shader_loader.h
 *  @brief  Some pretty basic tools for OGL rendering
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef SHADER_LOADER_H_INCLUDED
#define SHADER_LOADER_H_INCLUDED


#include <GL/glx.h>
#include <GL/gl.h>
//#include <GL/glext.h>


/**
* @brief The structure that defines what a Shader Consists of
*/
struct shaderObject
{
 // GLuint vertexShader, fragmentShader;
  int ProgramObject;
  int vertexShaderObject;
  int fragmentShaderObject;

  char * vertMem;
  unsigned long vertMemLength;

  char * fragMem;
  unsigned long fragMemLength;
};


char * loadShaderFileToMem(char * filename,unsigned long * file_length);

int useShader(struct shaderObject * shader);





/**
* @brief Load a pair of vertex and fragment shaders and populate a shader structure
* @ingroup OGLShaders
* @param String of filename of the file to load
* @param Output Integer that will say how big the memory chunk loaded is
* @retval 0=Error , A pointer to the block of memory with contents from filename
*/
struct shaderObject * loadShader(char * vertexShaderChar,char * fragmentShaderChar);



/**
* @brief Unload a pair of vertex and fragment shaders
* @ingroup OGLShaders
* @param A struct shaderObject
* @retval 0=Error , 1=Success
*/
int unloadShader(struct shaderObject * sh);

#endif // SHADER_LOADER_H_INCLUDED
