/** @file ogl_rendering.h
 *  @brief  This is file should handle all the OpenGL drawing and be able to switch graphics output to use fixed or shader based pipelines
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef OGL_RENDERING_H_INCLUDED
#define OGL_RENDERING_H_INCLUDED


//Master Switch for lighting..
#define USE_LIGHTS 1


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
int useLighting;
float lightPos[3];
//--------------------------------------

int doCulling;
};



int resetRendererOptions();



/**
* @brief Before starting to render something using OGL we need to call this function to initialize everything
* @ingroup Rendering
*/
int startOGLRendering();

int renderOGLLight( float * pos , unsigned int * parentNode ,  unsigned int boneSizes);

int renderOGLBones(
                 float * pos ,
                 unsigned int * parentNode ,
                 unsigned int boneSizes
                );

int renderOGL(
               const float * projectionMatrix ,
               const float * viewMatrix ,
               const float * modelMatrix ,
               const float * mvpMatrix ,
               //-------------------------------------------------------
               const float * vertices ,       unsigned int numberOfVertices ,
               const float * normal ,         unsigned int numberOfNormals ,
               const float * textureCoords ,  unsigned int numberOfTextureCoords ,
               const float * colors ,         unsigned int numberOfColors ,
               const unsigned int * indices , unsigned int numberOfIndices
             );

/**
* @brief After rendering using OGL we need to call this function to clean up
* @ingroup Rendering
*/
int stopOGLRendering();

#endif // OGL_RENDERING_H_INCLUDED
