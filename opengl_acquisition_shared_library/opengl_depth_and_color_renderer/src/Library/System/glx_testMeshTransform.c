#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Include GLEW
#include <GL/glew.h>

//GLU
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <time.h>
#include <string.h>

#include "../Tools/save_to_file.h"
#include "../Tools/tools.h"

#include "../ModelLoader/hardcoded_shapes.h"
#include "../ModelLoader/model_loader_tri.h"
#include "../ModelLoader/model_loader_transform_joints.h"

#include "../ModelLoader/tri_bvh_controller.h"

#include "../MotionCaptureLoader/bvh_loader.h"
#include "../MotionCaptureLoader/edit/bvh_rename.h"
#include "../MotionCaptureLoader/edit/bvh_randomize.h"

#include "glx3.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"

#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"
#include "../Rendering/ShaderPipeline/uploadGeometry.h"
#include "../Rendering/ShaderPipeline/uploadTextures.h"
#include "../Rendering/downloadFromRenderer.h"

//Colored console output..
#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//Change this to change MultiRendering numbers..
#define originalWIDTH 1080
#define originalHEIGHT 1080
#define tilesToDoX 1
#define tilesToDoY 1
#define shrinkingFactor 1
//--------------------------------------------

float lastFramerate = 60;
unsigned long lastRenderingTime = 0;
unsigned int framesRendered = 0;

char renderHair = 1;

//Virtual Camera Intrinsics
float fX = 1235.423889;
float fY = 1233.48468;
float nearPlane = 0.1;
float farPlane  = 1000.0;


struct pose6D
 {
     float x,y,z;
     float roll,pitch,yaw;

     char usePoseMatrixDirectly;
     struct Matrix4x4OfFloats m;
 };

int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}

int drawObjectAT(
                 GLuint programID,
                 GLuint vao,
                 GLuint MatrixID,
                 GLuint TextureID,
                 unsigned int triangleCount,
                 unsigned int elementCount,
                 //-----------------------------------------
                 float x,
                 float y,
                 float z,
                 float roll,
                 float pitch,
                 float yaw,
                 //-----------------------------------------
                 struct Matrix4x4OfFloats * projectionMatrix,
                 struct Matrix4x4OfFloats * viewportMatrix,
                 struct Matrix4x4OfFloats * viewMatrix,
                 //-----------------------------------------
                 char renderWireframe
                 )
{
       struct Matrix4x4OfFloats modelMatrix;
       create4x4FModelTransformation(
                                     &modelMatrix,
                                     //Rotation Component
                                     roll,//roll
                                     pitch ,//pitch
                                     yaw ,//yaw
                                     ROTATION_ORDER_RPY,
                                     //-----------------------------------------
                                     //Translation Component (XYZ)
                                     (float) x,
                                     (float) y,
                                     (float) z,
                                     //-----------------------------------------
                                     -1.0,//scaleX,
                                     1.0,//scaleY,
                                     -1.0//scaleZ
                                    );

       return drawVertexArrayWithMVPMatrices(
                                             programID,
                                             vao,
                                             MatrixID,
                                             TextureID,
                                             triangleCount,
                                             elementCount,
                                             //-----------------------------------------
                                             &modelMatrix,
                                             //-----------------------------------------
                                             projectionMatrix,
                                             viewportMatrix,
                                             viewMatrix,
                                             //-----------------------------------------
                                             renderWireframe
                                            );
}




int doOGLSingleDrawing(
                        int programID,
                        GLuint MVPMatrixID,
                        GLuint TextureID,
                        struct pose6D * pose,
                        GLuint VAO,
                        unsigned int triangleCount,
                        unsigned int elementCount,
                        unsigned int width,
                        unsigned int height
                      )
{
  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewportMatrix;
  struct Matrix4x4OfFloats viewMatrix;

  prepareRenderingMatrices(
                              fX, //fx
                              fY, //fy
                              0.0,        //skew
                              (float) width/2,    //cx
                              (float) height/2,   //cy
                              (float) width,      //Window Width
                              (float) height,     //Window Height
                              nearPlane,     //Near
                              farPlane,      //Far
                              &projectionMatrix,
                              &viewMatrix,
                              &viewportMatrix
                         );

  unsigned int viewportWidth = (unsigned int) width;
  unsigned int viewportHeight = (unsigned int) height;
  glViewport(0,0,viewportWidth,viewportHeight);

  float newViewport[4]={0,0,viewportWidth,viewportHeight};
  float projectionMatrixViewportCorrected[16];
  correctProjectionMatrixForDifferentViewport(
                                               projectionMatrixViewportCorrected,
                                               projectionMatrix.m,
                                               viewportMatrix.m,
                                               newViewport
                                             );
  //-------------------------------
  //-------------------------------
  //-------------------------------
  if (pose->usePoseMatrixDirectly)
     {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     VAO,
                                     MVPMatrixID,
                                     TextureID,
                                     triangleCount,
                                     elementCount,
                                     //-------------
                                     &pose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
     } else
     {
      drawObjectAT(
                  programID,
                  VAO,
                  MVPMatrixID,
                  TextureID,
                  triangleCount,
                  elementCount,
                  //-------------
                  pose->x,
                  pose->y,
                  pose->z,
                  pose->roll,
                  pose->pitch,
                  pose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
     }
  //-------------------------------
  //-------------------------------
  //-------------------------------

  return 1;
}






int doOGLDrawing(
                 int programID,
                 GLuint MVPMatrixID,
                 //------------------
                 GLuint eyelashesVao,
                 unsigned int eyelashesTriangleCount,
                 unsigned int eyelashesElementCount,
                 GLuint eyelashesTextureID,
                 //------------------
                 GLuint eyebrowsVao,
                 unsigned int eyebrowsTriangleCount,
                 unsigned int eyebrowsElementCount,
                 GLuint eyebrowsTextureID,
                 //------------------
                 GLuint hairVao,
                 unsigned int hairTriangleCount,
                 unsigned int hairElementCount,
                 GLuint hairTextureID,
                 //------------------
                 GLuint eyeVao,
                 unsigned int eyeTriangleCount,
                 unsigned int eyeElementCount,
                 GLuint eyeTextureID,
                 //------------------
                 struct pose6D * humanPose,
                 //------------------
                 GLuint humanVao,
                 unsigned int humanTriangleCount,
                 unsigned int humanElementCount,
                 GLuint humanTextureID,
                 //------------------
                 unsigned int width,
                 unsigned int height
                )
{
  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewportMatrix;
  struct Matrix4x4OfFloats viewMatrix;

  prepareRenderingMatrices(
                              fX,  //fx
                              fY,  //fy
                              0.0,        //skew
                              (float) width/2,    //cx
                              (float) height/2,   //cy
                              (float) width,      //Window Width
                              (float) height,     //Window Height
                              nearPlane,     //Near
                              farPlane,      //Far
                              &projectionMatrix,
                              &viewMatrix,
                              &viewportMatrix
                         );

  unsigned int viewportWidth = (unsigned int) width;
  unsigned int viewportHeight = (unsigned int) height;
  glViewport(0,0,viewportWidth,viewportHeight);

  float newViewport[4]={0,0,viewportWidth,viewportHeight};
  float projectionMatrixViewportCorrected[16];
  correctProjectionMatrixForDifferentViewport(
                                               projectionMatrixViewportCorrected,
                                               projectionMatrix.m,
                                               viewportMatrix.m,
                                               newViewport
                                             );
  //-------------------------------
  //-------------------------------
  //-------------------------------
  if (humanPose->usePoseMatrixDirectly)
     {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     eyelashesVao,
                                     MVPMatrixID,
                                     eyelashesTextureID,
                                     eyelashesTriangleCount,
                                     eyelashesElementCount,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     eyebrowsVao,
                                     MVPMatrixID,
                                     eyebrowsTextureID,
                                     eyebrowsTriangleCount,
                                     eyebrowsElementCount,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      if (renderHair)
      {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     hairVao,
                                     MVPMatrixID,
                                     hairTextureID,
                                     hairTriangleCount,
                                     hairElementCount,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      }
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     eyeVao,
                                     MVPMatrixID,
                                     eyeTextureID,
                                     eyeTriangleCount,
                                     eyeElementCount,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     humanVao,
                                     MVPMatrixID,
                                     humanTextureID,
                                     humanTriangleCount,
                                     humanElementCount,
                                     //-------------
                                     &humanPose->m,
                                     //-------------
                                     &projectionMatrix,
                                     &viewportMatrix,
                                     &viewMatrix,
                                     0 //Wireframe
                                    );
     } else
     {
      drawObjectAT(
                  programID,
                  eyelashesVao,
                  MVPMatrixID,
                  eyelashesTextureID,
                  eyelashesTriangleCount,
                  eyelashesElementCount,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      drawObjectAT(
                  programID,
                  eyebrowsVao,
                  MVPMatrixID,
                  eyebrowsTextureID,
                  eyebrowsTriangleCount,
                  eyebrowsElementCount,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      if (renderHair)
      {
      drawObjectAT(
                  programID,
                  hairVao,
                  MVPMatrixID,
                  hairTextureID,
                  hairTriangleCount,
                  hairElementCount,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      }
      drawObjectAT(
                  programID,
                  eyeVao,
                  MVPMatrixID,
                  eyeTextureID,
                  eyeTriangleCount,
                  eyeElementCount,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
      drawObjectAT(
                  programID,
                  humanVao,
                  MVPMatrixID,
                  humanTextureID,
                  humanTriangleCount,
                  humanElementCount,
                  //-------------
                  humanPose->x,
                  humanPose->y,
                  humanPose->z,
                  humanPose->roll,
                  humanPose->pitch,
                  humanPose->yaw,
                  //-------------
                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix,
                  0 //Wireframe
                  );
     }
  //-------------------------------
  //-------------------------------
  //-------------------------------

  return 1;
}

int initializeOGLRenderer(
                           GLuint * programID,
                           GLuint * programFrameBufferID,
                           GLuint * FramebufferName,
                           GLuint * renderedTexture,
                           GLuint * renderedDepth,
                           unsigned int WIDTH,
                           unsigned int HEIGHT
                         )
{
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../../shaders/TransformVertexShader.vertexshader", "../../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../../shaders/simpleWithTexture.vert", "../../../shaders/simpleWithTexture.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); return 0; }

	struct shaderObject * textureFramebuffer = loadShader("../../../shaders/virtualFramebuffer.vert", "../../../shaders/virtualFramebuffer.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); return 0; }

    *programID = sho->ProgramObject;
    *programFrameBufferID = textureFramebuffer->ProgramObject;

	// Use our shader
	glUseProgram(*programID);

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    initializeFramebuffer(FramebufferName,renderedTexture,renderedDepth,WIDTH,HEIGHT);

    fprintf(stderr,"Ready to start pushing geometry  ");
    return 1;
}


int doDrawing(
                GLuint programID,
                GLuint programFrameBufferID,
                GLuint FramebufferName,
                GLuint renderedTexture,
                GLuint renderedDepth,
                struct pose6D * humanPose,
                struct TRI_Model * humanModel,
                struct TRI_Model * eyeModel,
                struct TRI_Model * hairModel,
                struct TRI_Model * eyebrowsModel,
                struct TRI_Model * eyelashesModel,
                unsigned int WIDTH,
                unsigned int HEIGHT,
                int renderForever
             )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint eyelashesVAO=0;
    GLuint eyelashesArrayBuffer=0;
    GLuint eyelashesElementBuffer=0;
    unsigned int eyelashesTriangleCount  =  (unsigned int)  eyelashesModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &eyelashesVAO,
                             &eyelashesArrayBuffer,
                             &eyelashesElementBuffer,
                             programID,
                             eyelashesModel->vertices       ,  eyelashesModel->header.numberOfVertices      * sizeof(float),
                             eyelashesModel->normal         ,  eyelashesModel->header.numberOfNormals       * sizeof(float),
                             eyelashesModel->textureCoords  ,  eyelashesModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             eyelashesModel->colors         ,  eyelashesModel->header.numberOfColors        * sizeof(float),
                             eyelashesModel->indices        ,  eyelashesModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
    GLuint eyebrowsVAO=0;
    GLuint eyebrowsArrayBuffer=0;
    GLuint eyebrowsElementBuffer=0;
    unsigned int eyebrowsTriangleCount  =  (unsigned int)  eyebrowsModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &eyebrowsVAO,
                             &eyebrowsArrayBuffer,
                             &eyebrowsElementBuffer,
                             programID,
                             eyebrowsModel->vertices       ,  eyebrowsModel->header.numberOfVertices      * sizeof(float),
                             eyebrowsModel->normal         ,  eyebrowsModel->header.numberOfNormals       * sizeof(float),
                             eyebrowsModel->textureCoords  ,  eyebrowsModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             eyebrowsModel->colors         ,  eyebrowsModel->header.numberOfColors        * sizeof(float),
                             eyebrowsModel->indices        ,  eyebrowsModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
    GLuint hairVAO=0;
    GLuint hairArrayBuffer=0;
    GLuint hairElementBuffer=0;
    unsigned int hairTriangleCount  =  (unsigned int)  hairModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &hairVAO,
                             &hairArrayBuffer,
                             &hairElementBuffer,
                             programID,
                             hairModel->vertices       ,  hairModel->header.numberOfVertices      * sizeof(float),
                             hairModel->normal         ,  hairModel->header.numberOfNormals       * sizeof(float),
                             hairModel->textureCoords  ,  hairModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             hairModel->colors         ,  hairModel->header.numberOfColors        * sizeof(float),
                             hairModel->indices        ,  hairModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
    GLuint eyeVAO=0;
    GLuint eyeArrayBuffer=0;
    GLuint eyeElementBuffer=0;
    unsigned int eyeTriangleCount  =  (unsigned int)  eyeModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &eyeVAO,
                             &eyeArrayBuffer,
                             &eyeElementBuffer,
                             programID,
                             eyeModel->vertices       ,  eyeModel->header.numberOfVertices      * sizeof(float),
                             eyeModel->normal         ,  eyeModel->header.numberOfNormals       * sizeof(float),
                             eyeModel->textureCoords  ,  eyeModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             eyeModel->colors         ,  eyeModel->header.numberOfColors        * sizeof(float),
                             eyeModel->indices        ,  eyeModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
    GLuint humanVAO=0;
    GLuint humanArrayBuffer=0;
    GLuint humanElementBuffer=0;
    unsigned int humanTriangleCount  =  (unsigned int)  humanModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &humanVAO,
                             &humanArrayBuffer,
                             &humanElementBuffer,
                             programID,
                             humanModel->vertices       ,  humanModel->header.numberOfVertices      * sizeof(float),
                             humanModel->normal         ,  humanModel->header.numberOfNormals       * sizeof(float),
                             humanModel->textureCoords  ,  humanModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             humanModel->colors         ,  humanModel->header.numberOfColors        * sizeof(float),
                             humanModel->indices        ,  humanModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer=0;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");

	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        //glClearColor(0.0,0.0,0.0,1);
        glClearColor(0.2,0.2,0.2,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen

        doOGLDrawing(
                     programID,
                     MVPMatrixID,
                     //------------------------
                     eyelashesVAO,
                     eyelashesTriangleCount,
                     eyelashesModel->header.numberOfIndices,
                     eyelashesModel->header.textureBindGLBuffer,
                     //------------------------
                     eyebrowsVAO,
                     eyebrowsTriangleCount,
                     eyebrowsModel->header.numberOfIndices,
                     eyebrowsModel->header.textureBindGLBuffer,
                     //------------------------
                     hairVAO,
                     hairTriangleCount,
                     hairModel->header.numberOfIndices,
                     hairModel->header.textureBindGLBuffer,
                     //------------------------
                     eyeVAO,
                     eyeTriangleCount,
                     eyeModel->header.numberOfIndices,
                     eyeModel->header.textureBindGLBuffer,
                     //------------------------
                     humanPose,
                     //------------------------
                     humanVAO,
                     humanTriangleCount,
                     humanModel->header.numberOfIndices,
                     humanModel->header.textureBindGLBuffer,
                     //------------------------
                     WIDTH,
                     HEIGHT
                    );

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1000); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	//-------------------------------------
	glDeleteBuffers(1, &eyelashesArrayBuffer);
	glDeleteBuffers(1, &eyebrowsArrayBuffer);
	glDeleteBuffers(1, &hairArrayBuffer);
	glDeleteBuffers(1, &humanArrayBuffer);
	glDeleteBuffers(1, &eyeArrayBuffer);
	//-------------------------------------
	glDeleteBuffers(1, &eyelashesElementBuffer);
	glDeleteBuffers(1, &eyebrowsElementBuffer);
	glDeleteBuffers(1, &hairElementBuffer);
	glDeleteBuffers(1, &humanElementBuffer);
	glDeleteBuffers(1, &eyeElementBuffer);
	//-------------------------------------
	glDeleteVertexArrays(1, &eyelashesVAO);
	glDeleteVertexArrays(1, &eyebrowsVAO);
	glDeleteVertexArrays(1, &hairVAO);
	glDeleteVertexArrays(1, &humanVAO);
	glDeleteVertexArrays(1, &eyeVAO);
	return 1;
}





int doSkeletonDraw(
                   GLuint programID,
                   GLuint programFrameBufferID,
                   GLuint FramebufferName,
                   GLuint renderedTexture,
                   GLuint renderedDepth,
                   struct pose6D * humanPose,
                   struct TRI_Model * humanModel,
                   struct TRI_Model * axisModel,
                   unsigned int WIDTH,
                   unsigned int HEIGHT,
                   int renderForever
                  )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint axisVAO=0;
    GLuint axisArrayBuffer=0;
    GLuint axisElementBuffer=0;
    unsigned int axisTriangleCount=0;

    int usePrimitive = 0;

    if (!usePrimitive)
    {
     axisTriangleCount  =  (unsigned int)  axisModel->header.numberOfVertices/3;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             axisModel->vertices       ,  axisModel->header.numberOfVertices      * sizeof(float),
                             axisModel->normal         ,  axisModel->header.numberOfNormals       * sizeof(float),
                             axisModel->textureCoords  ,  axisModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             axisModel->colors         ,  axisModel->header.numberOfColors        * sizeof(float),
                             axisModel->indices        ,  axisModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                           );
    } else
    {
     axisTriangleCount  = pyramidTriangleCount;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             pyramidCoords     ,  sizeof(pyramidCoords),
                             pyramidNormals    ,  sizeof(pyramidNormals),
                             pyramidTexCoords  ,  sizeof(pyramidTexCoords),      //0,0 //No Texture
                             cubeColors        ,  sizeof(cubeColors),
                             0                 ,  0//0,0 //Not Indexed
                          );
    }
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");
	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        glClearColor( 0, 0.0, 0, 1 );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen


        if (humanModel->bones==0)
        {
            fprintf(stderr,"doSkeletonDraw called with no bones!\n");
            return 0;
        }

          //Add an axis to help
          /*struct pose6D axisPose={0};
          axisPose.z=10;
          doOGLSingleDrawing(
                             programID,
                             MVPMatrixID,
                             0,
                             &axisPose,
                             axisVAO,
                             axisTriangleCount,
                             axisElementBuffer,
                             WIDTH,
                             HEIGHT
                           );*/


        fprintf(stderr,"BoneID %u -> %u \n",0,humanModel->header.numberOfBones);
        for (unsigned int boneID=0; boneID<humanModel->header.numberOfBones; boneID++)
        {
         if (humanModel->bones[boneID].info!=0) //humanPose->x =
         {
          //fprintf(stderr,GREEN "BoneID %u  \n" NORMAL,boneID);
          struct pose6D axisPose={0};
          axisPose.x = humanPose->x + humanModel->bones[boneID].info->x;
          axisPose.y = humanPose->y + humanModel->bones[boneID].info->y;
          axisPose.z = humanPose->z + humanModel->bones[boneID].info->z;

          axisPose.usePoseMatrixDirectly = 1;
          for (unsigned int i=0; i<16; i++)
          {
            axisPose.m.m[i] = humanModel->bones[boneID].info->finalVertexTransformation[i];
          }
          axisPose.m.m[3]  = 10 * axisPose.x; //X
          axisPose.m.m[7]  = 10 * axisPose.y; //Y
          axisPose.m.m[11] = 10 * axisPose.z; //Z

          //axisPose.m.m[3]  = 10 *axisPose.y; //X
          axisPose.m.m[7]  -= 10;  //Y
          axisPose.m.m[11] -= 100; //Z

          if (
               (strcmp("scene",humanModel->bones[boneID].boneName)!=0)   &&
               (strcmp("camera",humanModel->bones[boneID].boneName)!=0)  &&
               (strcmp("light",humanModel->bones[boneID].boneName)!=0)   &&
               (strcmp("testexp",humanModel->bones[boneID].boneName)!=0) &&
               (strcmp("test",humanModel->bones[boneID].boneName)!=0)
             )
          {
           //Only draw axis of parts of actual geometry
           doOGLSingleDrawing(
                              programID,
                              MVPMatrixID,
                              0,
                              &axisPose,
                              axisVAO,
                              axisTriangleCount,
                              axisElementBuffer,
                              WIDTH,
                              HEIGHT
                            );
          }
         } else
         {
           fprintf(stderr,RED "BoneID %u empty! \n" NORMAL,boneID);
         }
        }

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1000); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteBuffers(1, &axisArrayBuffer);
	glDeleteBuffers(1, &axisElementBuffer);
	glDeleteVertexArrays(1, &axisVAO);
	return 1;
}






int doBVHDraw(
               GLuint programID,
               GLuint programFrameBufferID,
               GLuint FramebufferName,
               GLuint renderedTexture,
               GLuint renderedDepth,
               struct pose6D * humanPose,
               struct BVH_MotionCapture * bvh,
               unsigned int frameID,
               struct TRI_Model * axisModel,
               unsigned int WIDTH,
               unsigned int HEIGHT,
               int renderForever
             )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint axisVAO=0;
    GLuint axisArrayBuffer=0;
    GLuint axisElementBuffer=0;
    unsigned int axisTriangleCount=0;

    int usePrimitive = 0;

    if (!usePrimitive)
    {
     axisTriangleCount  =  (unsigned int)  axisModel->header.numberOfVertices/3;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             axisModel->vertices       ,  axisModel->header.numberOfVertices      * sizeof(float),
                             axisModel->normal         ,  axisModel->header.numberOfNormals       * sizeof(float),
                             axisModel->textureCoords  ,  axisModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             axisModel->colors         ,  axisModel->header.numberOfColors        * sizeof(float),
                             axisModel->indices        ,  axisModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                           );
    } else
    {
     axisTriangleCount  = pyramidTriangleCount;
     pushObjectToBufferData(
                             1,
                             &axisVAO,
                             &axisArrayBuffer,
                             &axisElementBuffer,
                             programID,
                             pyramidCoords     ,  sizeof(pyramidCoords),
                             pyramidNormals    ,  sizeof(pyramidNormals),
                             pyramidTexCoords  ,  sizeof(pyramidTexCoords),      //0,0 //No Texture
                             cubeColors        ,  sizeof(cubeColors),
                             0        ,  0//0,0 //Not Indexed
                          );
    }
    //------------------------------------------------------------------------------------
	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");

	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	 do
     {
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        glClearColor( 0, 0.0, 0, 1 );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen


        struct BVH_Transform bvhTransform= {0};
        if (
                bvh_loadTransformForFrame(
                                           bvh,
                                           frameID,
                                           &bvhTransform,
                                           0
                                         )
             )
          {
           for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
            {
               struct pose6D axisPose={0};
               axisPose.x = bvhTransform.joint[jID].pos3D[0]/10;
               axisPose.y = bvhTransform.joint[jID].pos3D[1]/10;
               axisPose.z = 50+bvhTransform.joint[jID].pos3D[2]/10;

               /*
                 struct Matrix4x4OfFloats localToWorldTransformation;
                 struct Matrix4x4OfFloats chainTransformation;
                 struct Matrix4x4OfFloats dynamicTranslation;
                 struct Matrix4x4OfFloats dynamicRotation; */
               axisPose.usePoseMatrixDirectly = 1;
               for (unsigned int i=0; i<16; i++)
                 {
                   axisPose.m.m[i] = bvhTransform.joint[jID].localToWorldTransformation.m[i];
                 }
               axisPose.m.m[3]  = axisPose.m.m[3]/10;
               axisPose.m.m[7]  = axisPose.m.m[7]/10;
               axisPose.m.m[11] = 50 + axisPose.m.m[11]/10 ; //Send rendering 50 units away..

               doOGLSingleDrawing(
                                   programID,
                                   MVPMatrixID,
                                   0,
                                   &axisPose,
                                   axisVAO,
                                   axisTriangleCount,
                                   axisElementBuffer,
                                   WIDTH,
                                   HEIGHT
                                 );
            }
          }

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();


      //---------------------------------------------------------------
      //------------------- Calculate Framerate -----------------------
      //---------------------------------------------------------------
       unsigned long now=GetTickCountMilliseconds();
       unsigned long elapsedTime=now-lastRenderingTime;
       if (elapsedTime==0) { elapsedTime=1; }
       lastFramerate = (float) 1000/(elapsedTime);
       lastRenderingTime = now;
      //---------------------------------------------------------------
        ++framesRendered;
      //---------------------------------------------------------------

      if (renderForever) { usleep(1000); } // Cap framerate if looping here...
	} // Check if the ESC key was pressed or the window was closed
    while(renderForever);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &quad_vertexbuffer);
	glDeleteBuffers(1, &axisArrayBuffer);
	glDeleteBuffers(1, &axisElementBuffer);
	glDeleteVertexArrays(1, &axisVAO);
	return 1;
}




 /*
 location&rotation
 oculi01.l/r
 risorius03.l/r
 levator06.l/r
 oris03.l/r
 oris05
 levator05.l/r
 oris07.l/r
 oris01

 rotation
 orbicularis03.l/r
 orbicularis04.l/r
 eye.l/r */
void randomizeHead(struct BVH_MotionCapture * mc)
{
   BVHJointID jIDLeft;
   BVHJointID jIDRight;
   float r = 0.0;

   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"neck",&jIDLeft))
      )
    {
      r = randomFloatA(-60,60);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
    }


   //                            ROTATION ONLY
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis03.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis04.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"orbicularis04.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"eye.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"eye.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);

      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
    }




   //                         LOCATION &  ROTATION ONLY
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oculi01.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oculi01.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"risorius03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"risorius03.r",&jIDRight))
      )
    {
      r = randomFloatA(-50,50);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"levator06.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"levator06.r",&jIDRight))
      )
    {
      r = randomFloatA(-20,20);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-10,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,-r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris03.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oris03.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationXAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris05",&jIDLeft))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris01",&jIDLeft))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      r = randomFloatA(-30,30);
      bvh_setJointRotationXAtFrame(mc,jIDLeft,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"levator05.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"levator05.r",&jIDRight))
      )
    {
      r = randomFloatA(-60,60);
      bvh_setJointRotationYAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationYAtFrame(mc,jIDRight,0,r);
    }
   //===========================================================================
   //===========================================================================
   //===========================================================================
   if (
        (bvh_getJointIDFromJointNameNocase(mc,"oris07.l",&jIDLeft)) &&
        (bvh_getJointIDFromJointNameNocase(mc,"oris07.r",&jIDRight))
      )
    {
      r = randomFloatA(-30,30);
      bvh_setJointRotationZAtFrame(mc,jIDLeft,0,r);
      bvh_setJointRotationZAtFrame(mc,jIDRight,0,r);
    }
}





int main(int argc,const char **argv)
{
  unsigned int WIDTH =(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
  unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;

  fprintf(stderr,"Attempting to setup a %ux%u glx3 context\n",WIDTH,HEIGHT);
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }

  GLuint programID=0;
  GLuint programFrameBufferID=0;
  GLuint FramebufferName=0;
  GLuint renderedTexture=0;
  GLuint renderedDepth=0;

  if (!initializeOGLRenderer(&programID,&programFrameBufferID,&FramebufferName,&renderedTexture,&renderedDepth,WIDTH,HEIGHT))
  {
		fprintf(stderr, "Failed to initialize Shaders\n");
	 	return 1;
  }

   #define defaulBVHToLoad "01_02.bvh" //merged_neutral.bvh
   #define defaultModelToLoad "makehuman.tri"

   const char * bvhToLoad = defaulBVHToLoad;
   const char * modelToLoad = defaultModelToLoad;

   //------------------------------------------------------
   struct BVH_MotionCapture mc = {0};
   //------------------------------------------------------
   struct TRI_Model axisModel={0};
   //------------------------------------------------------
   struct TRI_Model eyeModel={0};
   struct TRI_Model indexedEyeModel={0};
   //------------------------------------------------------
   struct TRI_Model hairModel={0};
   struct TRI_Model indexedHairModel={0};
   //------------------------------------------------------
   struct TRI_Model eyebrowsModel={0};
   struct TRI_Model indexedEyebrowsModel={0};
   //------------------------------------------------------
   struct TRI_Model eyelashesModel={0};
   struct TRI_Model indexedEyelashesModel={0};
   //------------------------------------------------------
   struct pose6D humanPose={0};
   struct TRI_Model humanModel={0};
   struct TRI_Model indexedHumanModel={0};
   //------------------------------------------------------
   int destroyColors=0;
   int dumpVideo = 0;
   int dumpSnapshot = 0;
   unsigned int maxFrames=0;

   int axisRendering = 0;
   int staticRendering = 0;

   int randomize=0;

   /*
   //DAE output Set human pose to somewhere visible..
   //-------------------------------------------------------------------
   humanPose.roll=180.0;//(float)  (rand()%90);
   humanPose.pitch=0.0;//(float) (rand()%90);
   humanPose.yaw=0.0;//(float)   (rand()%90);
   //-------------------------------------------------------------------
   humanPose.x=0.0f;//(float)  (1000-rand()%2000);
   humanPose.y=-7.976f;//(float) (100-rand()%200);
   humanPose.z=27.99735f;//(float)  (700+rand()%1000);
   //-------------------------------------------------------------------
*/

   //MHX2 Set human pose to somewhere visible..
   //-------------------------------------------------------------------
   humanPose.roll=0.0;//180.0;//(float)  (rand()%90);
   humanPose.pitch=90.0;//-90.0;//(float) (rand()%90);
   humanPose.yaw=0.0;//(float)   (rand()%90);
   //-------------------------------------------------------------------
   humanPose.x=0.0f;//(float)  (1000-rand()%2000);
   humanPose.y=-0.976f;//(float) (100-rand()%200);
   humanPose.z=2.99735f;//(float)  (700+rand()%1000);
   //-------------------------------------------------------------------

   //------------------------------------------------------
   for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--axis")==0)
                    {
                      axisRendering=1;
                    } else
           if (strcmp(argv[i],"--bvhaxis")==0)
                    {
                      axisRendering=2;
                    } else
           if (strcmp(argv[i],"--static")==0)
                    {
                      staticRendering=1;
                    } else
           if (strcmp(argv[i],"--face")==0)
                    {
                       //  ./gl3MeshTransform --face --set eye.l x 20 --set eye.r x 20 --set eye.l z 20 --set eye.r z 20 --set orbicularis03.l x 30 --set orbicularis03.r x 30
                       //  ./gl3MeshTransform --face --set eye.l x 20 --set eye.r x 20 --set eye.l z 20 --set eye.r z 20
                       //  ./gl3MeshTransform --face --bvh merged_neutral.bvh

                       //Regular DAE align global=local
                       humanPose.x=0.0f;//(float)  (1000-rand()%2000);
                       humanPose.y=-14.976f;//(float) (100-rand()%200);
                       humanPose.z=7.99735f;//(float)  (700+rand()%1000);

                       //MHX2
                       humanPose.x=0.0f;//(float)  (1000-rand()%2000);
                       humanPose.y=-1.476f;//(float) (100-rand()%200);
                       humanPose.z=0.69735f;//(float)  (700+rand()%1000);
                    } else
           if (strcmp(argv[i],"--save")==0)
                    {
                      dumpSnapshot=1;
                    } else
           if (strcmp(argv[i],"--dumpvideo")==0)
                    {
                      dumpVideo=1;
                    } else
           if (strcmp(argv[i],"--nocolor")==0)
                    {
                      destroyColors=1;
                    } else
           if (strcmp(argv[i],"--bvh")==0)
                    {
                        if (argc>i+1)
                            {
                                bvhToLoad = argv[i+1];
                            }
                    } else
           if (strcmp(argv[i],"--from")==0)
                    {
                        if (argc>i+1)
                            {
                                modelToLoad = argv[i+1];
                            }
                    } else
           if (strcmp(argv[i],"--maxFrames")==0)
                    {
                        if (argc>i+1)
                            {
                                maxFrames = atoi(argv[i+1]);
                            }
                    }
        }
   //------------------------------------------------------
   if (!bvh_loadBVH(bvhToLoad,&mc,1.0) ) // This is the new armature that includes the head
        {
          fprintf(stderr,"Cannot find the merged_neutral.bvh file..\n");
          return 0;
        }
   bvh_renameJointsToLowercase(&mc);
   //------------------------------------------------------
   if (axisRendering)
   {
     struct TRI_Model axisModelIndexed={0};
     if (!tri_loadModel("axis.tri", &axisModelIndexed ) )
     {
       fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/axis.tri\n");
       return 0;
     }
     //paintTRI(&axisModelIndexed,123,123,123);
     tri_flattenIndexedModel(&axisModel,&axisModelIndexed);
   }
   //------------------------------------------------------
   if (!tri_loadModel(modelToLoad, &indexedHumanModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/makehuman.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyes.tri", &indexedEyeModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyes.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("hair.tri", &indexedHairModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/hair.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyebrows.tri", &indexedEyebrowsModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyebrows.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!tri_loadModel("eyelashes.tri", &indexedEyelashesModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyelashes.tri\n");
     return 0;
   }

   if (destroyColors)
   {
      tri_paintModel(&indexedHumanModel,123,123,123);
      tri_paintModel(&indexedEyeModel,123,123,123);
      tri_paintModel(&indexedHairModel,123,123,123);
      tri_paintModel(&indexedEyebrowsModel,123,123,123);
      tri_paintModel(&indexedEyelashesModel,123,123,123);
   }
   //------------------------------------------------------

   unsigned char * rgb = 0;
   if ( (dumpVideo) || (dumpSnapshot) )
      { rgb =  (unsigned char * ) malloc(sizeof(unsigned char) * WIDTH * HEIGHT *3); }

   //------------------------------------------------------
   fprintf(stderr,"Preprocessing human model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedHumanModel);
   tri_removePrefixFromAllBoneNames(&indexedHumanModel,"test_"); //Eyes have a test_ prefix on bone names..
   tri_removePrefixFromAllBoneNames(&indexedHumanModel,"f_"); //Fingers have a f_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedHumanModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eye model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyeModel);
   tri_removePrefixFromAllBoneNames(&indexedEyeModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyeModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing hair model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedHairModel);
   tri_removePrefixFromAllBoneNames(&indexedHairModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedHairModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eyebrows model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyebrowsModel);
   tri_removePrefixFromAllBoneNames(&indexedEyebrowsModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyebrowsModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------
   fprintf(stderr,"Preprocessing eyelashes model.. ");
   tri_makeAllBoneNamesLowerCase(&indexedEyelashesModel);
   tri_removePrefixFromAllBoneNames(&indexedEyelashesModel,"test_"); //Eyes have a test_ prefix on bone names..
   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyelashesModel);
   fprintf(stderr,GREEN "OK\n" NORMAL);
   //------------------------------------------------------

   bvh_printBVH(&mc);
   printTRIBoneStructure(&indexedHumanModel,0 /*alsoPrintMatrices*/);


   if (indexedHumanModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedHumanModel.header.textureBindGLBuffer,
                                 &indexedHumanModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedHumanModel.textureData,
                                  indexedHumanModel.header.textureDataWidth,
                                  indexedHumanModel.header.textureDataHeight,
                                  indexedHumanModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedHumanModel.textureData);
      //indexedHumanModel.textureData=0;
   }

   if (indexedEyeModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyeModel.header.textureBindGLBuffer,
                                 &indexedEyeModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyeModel.textureData,
                                  indexedEyeModel.header.textureDataWidth,
                                  indexedEyeModel.header.textureDataHeight,
                                  indexedEyeModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyeModel.textureData);
      //indexedEyeModel.textureData=0;
   }

   if (indexedHairModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedHairModel.header.textureBindGLBuffer,
                                 &indexedHairModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedHairModel.textureData,
                                  indexedHairModel.header.textureDataWidth,
                                  indexedHairModel.header.textureDataHeight,
                                  indexedHairModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedHairModel.textureData);
      //indexedHairModel.textureData=0;
   }

   if (indexedEyebrowsModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyebrowsModel.header.textureBindGLBuffer,
                                 &indexedEyebrowsModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyebrowsModel.textureData,
                                  indexedEyebrowsModel.header.textureDataWidth,
                                  indexedEyebrowsModel.header.textureDataHeight,
                                  indexedEyebrowsModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyebrowsModel.textureData);
      //indexedEyebrowsModel.textureData=0;
   }


   if (indexedEyelashesModel.textureData!=0)
   {
      uploadColorImageAsTexture(
                                 programID,
                                 (GLuint *) &indexedEyelashesModel.header.textureBindGLBuffer,
                                 &indexedEyelashesModel.header.textureUploadedToGPU,
                                 (unsigned char*) indexedEyelashesModel.textureData,
                                  indexedEyelashesModel.header.textureDataWidth,
                                  indexedEyelashesModel.header.textureDataHeight,
                                  indexedEyelashesModel.header.textureDataChannels,
                                  24
                                );
      //Don't copy data around since its already on GPU
      //free(indexedEyelashesModel.textureData);
      //indexedEyelashesModel.textureData=0;
   }




  fprintf(stderr,"eyelashesTextureID = %u \n",indexedEyelashesModel.header.textureBindGLBuffer);
  fprintf(stderr,"eyebrowsTextureID = %u \n",indexedEyebrowsModel.header.textureBindGLBuffer);
  fprintf(stderr,"hairTextureID = %u \n",indexedHairModel.header.textureBindGLBuffer);
  fprintf(stderr,"eyeTextureID = %u \n",indexedEyeModel.header.textureBindGLBuffer);
  fprintf(stderr,"humanTextureID = %u \n",indexedHumanModel.header.textureBindGLBuffer);



   //------------------------------------------------------
   //  ./gl3MeshTransform --set relbow z 90 --set hip y 180 --set lelbow x 90 --set rknee z 45 --set lshoulder y -45
   //  ./gl3MeshTransform --bvhaxis --set relbow z 90 --set hip y 180 --set lelbow y 90 --set rknee z -45 --set lshoulder x 45
   for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--randomize")==0)
                    {
                       randomize=1;
                       mc.numberOfFrames=1;
                    } else
           if (strcmp(argv[i],"--set")==0)
                    {
                      const char * jointName = argv[i+1];
                      const char * jointAxis = argv[i+2];
                      float  jointValue = atof(argv[i+3]);
                      BVHJointID jID;

                      if (bvh_getJointIDFromJointNameNocase(&mc,jointName,&jID))
                      {
                          if (strcmp("x",jointAxis)==0)
                          {
                              bvh_setJointRotationXAtFrame(&mc,jID,0,jointValue);
                          } else
                          if (strcmp("y",jointAxis)==0)
                          {
                              bvh_setJointRotationYAtFrame(&mc,jID,0,jointValue);
                          } else
                          if (strcmp("z",jointAxis)==0)
                          {
                              bvh_setJointRotationZAtFrame(&mc,jID,0,jointValue);
                          }
                      }

                      mc.numberOfFrames=1;
                    }
        }
   //------------------------------------------------------


   if (maxFrames==0)
   {
       maxFrames = mc.numberOfFrames;
   }

   do
    {
      if (randomize)
        {
            randomizeHead(&mc);
            usleep(100000);
        }


    for (BVHFrameID fID=0; fID<maxFrames; fID++)
    {
     fprintf(stderr,CYAN "\nBVH %s Frame %u/%u (BVH has %u frames total) \n" NORMAL,mc.fileName,fID,maxFrames,mc.numberOfFrames);
     //-------------------------------------------

     if (!staticRendering)
     {
       animateTRIModelUsingBVHArmature(&humanModel,&indexedHumanModel,&mc,fID,0);
       //triDeepCopyBoneValuesButNotStructure(&indexedEyeModel,&indexedHumanModel);
       animateTRIModelUsingBVHArmature(&eyeModel,&indexedEyeModel,&mc,fID,0);
       animateTRIModelUsingBVHArmature(&hairModel,&indexedHairModel,&mc,fID,0);
       animateTRIModelUsingBVHArmature(&eyebrowsModel,&indexedEyebrowsModel,&mc,fID,0);
       animateTRIModelUsingBVHArmature(&eyelashesModel,&indexedEyelashesModel,&mc,fID,0);
     } else
     {
       tri_flattenIndexedModel(&humanModel,&indexedHumanModel);
       tri_flattenIndexedModel(&eyeModel,&indexedEyeModel);
       tri_flattenIndexedModel(&hairModel,&indexedHairModel);
       tri_flattenIndexedModel(&eyebrowsModel,&indexedEyebrowsModel);
       tri_flattenIndexedModel(&eyelashesModel,&indexedEyelashesModel);
     }


    if (axisRendering == 1)
    {
      humanPose.roll+=1.0;//(float)  (rand()%90);
      humanPose.pitch+=1.0;//(float) (rand()%90);
      humanPose.yaw+=1.0;//(float)   (rand()%90);
      //-------------------------------------------------------------------
      humanPose.x=0.0f;//(float)  (1000-rand()%2000);
      humanPose.y=0.0f;//(float) (100-rand()%200);
      humanPose.z=13.4f;//(float)  (700+rand()%1000);

     //Do axis rendering
     doSkeletonDraw(
                     programID,
                     programFrameBufferID,
                     FramebufferName,
                     renderedTexture,
                     renderedDepth,
                     &humanPose,
                     &indexedHumanModel,
                     &axisModel,
                     WIDTH,
                     HEIGHT,
                     0
                   );
    } else
    if (axisRendering == 2)
    {
      doBVHDraw(
                 programID,
                 programFrameBufferID,
                 FramebufferName,
                 renderedTexture,
                 renderedDepth,
                 &humanPose,
                 &mc,
                 fID,
                 &axisModel,
                 WIDTH,
                 HEIGHT,
                 0
               );
    } else
    { //Do regular skinned model rendering
     doDrawing(
                programID,
                programFrameBufferID,
                FramebufferName,
                renderedTexture,
                renderedDepth,
                &humanPose,
                &humanModel,
                &eyeModel,
                &hairModel,
                &eyebrowsModel,
                &eyelashesModel,
                WIDTH,
                HEIGHT,
                0
              );
    }


     tri_deallocModelInternals(&humanModel);
     tri_deallocModelInternals(&eyeModel);
     tri_deallocModelInternals(&hairModel);
     tri_deallocModelInternals(&eyebrowsModel);
     usleep(1);

     if  (rgb!=0)
     {
       if (downloadOpenGLColor(rgb,0,0,WIDTH,HEIGHT))
       {
           char filename[512]={0};
           snprintf(filename,512,"colorFrame_0_%05u.pnm",fID);
           saveRawImageToFileOGLR(filename,rgb,WIDTH,HEIGHT,3,8);
       }
     }

    }

      if (maxFrames>1)
        { fprintf(stderr,CYAN "\n\nLooping Dataset\n\n" NORMAL); }
   }
   while ( (dumpVideo==0) && (dumpSnapshot==0) ); //If dump video is not enabled loop forever


   glDeleteProgram(programID);

   //If we dumped images now lets convert them to video
   if (rgb!=0)
     {
        //Also deallocate our snapshot buffer..
        free(rgb);
        int i=0;

        if (dumpVideo)
         {
           char command[512]={0};
           snprintf(command,512,"ffmpeg -framerate 30 -i colorFrame_0_%%05d.pnm -s %ux%u -y -r 30 -pix_fmt yuv420p -threads 8 lastRun3DHiRes.mp4",WIDTH,HEIGHT);
           i=system(command);
         }

        if (dumpSnapshot==0)
        {
         if(i==0)
         {
          i=system("rm *.pnm");
         }
        }
     }


   stop_glx3_stuff();
 return 0;
}

