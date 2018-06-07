/** @file ogl_rendering.h
 *  @brief  This is file should handle all the OpenGL drawing and be able to switch graphics output to use fixed or shader based pipelines
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef OGL_RENDERING_H_INCLUDED
#define OGL_RENDERING_H_INCLUDED


#define MAX_SHADER_FILENAMES 512

struct rendererConfiguration
{
//Shader specific stuff ----------------
char fragmentShaderFile[MAX_SHADER_FILENAMES];
char * selectedFragmentShader;
char vertexShaderFile[MAX_SHADER_FILENAMES];
char * selectedVertexShader;
struct shaderObject * loadedShader;
int useShaders;
//--------------------------------------

int doCulling;
};



int resetRendererOptions();



/**
* @brief Before starting to render something using OGL we need to call this function to initialize everything
* @ingroup Rendering
*/
int startOGLRendering();



int renderOGLBones(
                 float * pos ,
                 unsigned int * parentNode ,
                 unsigned int boneSizes
                );

int renderOGL(
               float * projectionMatrix ,
               float * viewMatrix ,
               float * modelMatrix ,
               float * mvpMatrix ,
               //-------------------------------------------------------
               float * vertices ,       unsigned int numberOfVertices ,
               float * normal ,         unsigned int numberOfNormals ,
               float * textureCoords ,  unsigned int numberOfTextureCoords ,
               float * colors ,         unsigned int numberOfColors ,
               unsigned int * indices , unsigned int numberOfIndices
             );

/**
* @brief After rendering using OGL we need to call this function to clean up
* @ingroup Rendering
*/
int stopOGLRendering();

#endif // OGL_RENDERING_H_INCLUDED
