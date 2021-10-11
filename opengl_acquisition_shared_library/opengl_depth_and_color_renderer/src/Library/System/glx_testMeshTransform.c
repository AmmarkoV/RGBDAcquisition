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

#include "glx3.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"

#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"
#include "../Rendering/ShaderPipeline/uploadGeometry.h"



#define NORMAL  "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


//Change this to change MultiRendering numbers
#define originalWIDTH 1080
#define originalHEIGHT 1080
#define tilesToDoX 1
#define tilesToDoY 1
#define shrinkingFactor 1
//--------------------------------------------



unsigned int WIDTH=(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;

float lastFramerate = 60;
unsigned long lastRenderingTime = 0;
unsigned int framesRendered=0;


int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}





int drawObjectAT(GLuint programID,
                 GLuint vao,
                 GLuint MatrixID,
                 unsigned int triangleCount,

                 float x,
                 float y,
                 float z,
                 float roll,
                 float pitch,
                 float yaw,

                 struct Matrix4x4OfFloats * projectionMatrix,
                 struct Matrix4x4OfFloats * viewportMatrix,
                 struct Matrix4x4OfFloats * viewMatrix
                 )
{
       //Select Shader to render with
       glUseProgram(programID);                  checkOpenGLError(__FILE__, __LINE__);

       //Select Vertex Array Object To Render
       glBindVertexArray(vao);                   checkOpenGLError(__FILE__, __LINE__);

       //fprintf(stderr,"XYZRPY(%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f)\n",x,y,z,roll,pitch,yaw);


       struct Matrix4x4OfFloats modelMatrix;
       create4x4FModelTransformation(
                                    &modelMatrix,
                                    //Rotation Component
                                    roll,//roll
                                    pitch ,//pitch
                                    yaw ,//yaw
                                    ROTATION_ORDER_RPY,

                                    //Translation Component (XYZ)
                                    (float) x,
                                    (float) y,
                                    (float) z,

                                    1.0,//scaleX,
                                    1.0,//scaleY,
                                    1.0//scaleZ
                                   );

      //-------------------------------------------------------------------
       struct Matrix4x4OfFloats MVP;
       getModelViewProjectionMatrixFromMatrices(&MVP,projectionMatrix,viewMatrix,&modelMatrix);
       transpose4x4FMatrix(MVP.m);
      //-------------------------------------------------------------------



		// Send our transformation to the currently bound shader, in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE/*TRANSPOSE*/,MVP.m);



        glPushAttrib(GL_ALL_ATTRIB_BITS);
        //Our flipped view needs front culling..
        glCullFace(GL_FRONT);
        glEnable(GL_CULL_FACE);

         //-------------------------------------------------
         //if (wireFrame) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); else
         //               glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         // checkOpenGLError(__FILE__, __LINE__);
         //-------------------------------------------------


         glDrawArrays( GL_TRIANGLES, 0, triangleCount );   checkOpenGLError(__FILE__, __LINE__);


       glPopAttrib();
       glBindVertexArray(0); checkOpenGLError(__FILE__, __LINE__);
       return 1;
}



int doOGLDrawing(
                   int programID,
                   GLuint MVPMatrixID ,
                   GLuint eyeVao,
                   unsigned int eyeTriangleCount,
                   GLuint humanVao,
                   unsigned int humanTriangleCount,
                   unsigned int width,
                   unsigned int height
                   )
{
  struct Matrix4x4OfFloats projectionMatrix;
  struct Matrix4x4OfFloats viewportMatrix;
  struct Matrix4x4OfFloats viewMatrix;

  prepareRenderingMatrices(
                              1235.423889, //fx
                              1233.48468,  //fy
                              0.0,        //skew
                              (float) originalWIDTH/2,    //cx
                              (float) originalHEIGHT/2,   //cy
                              (float) originalWIDTH,      //Window Width
                              (float) originalHEIGHT,     //Window Height
                              1.0,        //Near
                              255.0,      //Far
                              &projectionMatrix,
                              &viewMatrix,
                              &viewportMatrix
                         );

  //-------------------------------------------------------------------
  float roll=180.0;//(float)  (rand()%90);
  float pitch=0.0;//(float) (rand()%90);
  float yaw=0.0;//(float)   (rand()%90);

  float x=0.0f;//(float)  (1000-rand()%2000);
  float y=-8.976f;//(float) (100-rand()%200);
  float z=26.99735f;//(float)  (700+rand()%1000);
  //-------------------------------------------------------------------

  unsigned int viewportWidth = (unsigned int) width;
  unsigned int viewportHeight = (unsigned int) height;

  glViewport(0,0, viewportWidth , viewportHeight );

  float newViewport[4]={0,0, viewportWidth , viewportHeight};
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
     //fprintf(stderr,"glViewport(%u,%u,%u,%u)\n",viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight);
     drawObjectAT(
                  programID,
                  eyeVao,
                  MVPMatrixID,
                  eyeTriangleCount,
                  x,
                  y+0.05,
                  z+0.8,
                  roll,
                  pitch,
                  yaw,

                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix
                 );
     //-------------------------------
     //-------------------------------
     //-------------------------------
     drawObjectAT(
                  programID,
                  humanVao,
                  MVPMatrixID,
                  humanTriangleCount,
                  x,
                  y,
                  z,
                  roll,
                  pitch,
                  yaw,

                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix
                 );
     //-------------------------------
     //-------------------------------
     //-------------------------------

      viewMatrix.m[3]+=1.0;

  return 1;
}





int doDrawing(struct TRI_Model * triModel , struct TRI_Model * eyeModel)
{
   fprintf(stderr," doDrawing \n");
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../../shaders/TransformVertexShader.vertexshader", "../../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../../shaders/simple.vert", "../../../shaders/simple.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

	struct shaderObject * textureFramebuffer = loadShader("../../../shaders/virtualFramebuffer.vert", "../../../shaders/virtualFramebuffer.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

    GLuint programID = sho->ProgramObject;
    GLuint programFrameBufferID = textureFramebuffer->ProgramObject;

 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");

	// Use our shader
	glUseProgram(programID);

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    fprintf(stderr,"Ready to start pushing geometry  ");

    GLuint eyeVAO;
    GLuint eyeArrayBuffer;
    unsigned int eyeTriangleCount  =  (unsigned int)  eyeModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &eyeVAO,
                             &eyeArrayBuffer,
                             programID  ,
                             eyeModel->vertices  ,  eyeModel->header.numberOfVertices * sizeof(float) ,
                             eyeModel->normal    ,  eyeModel->header.numberOfNormals  * sizeof(float),
                             0 ,  0, //No Texture
                             //eyeModel->textureCoords  ,  eyeModel->header.numberOfTextureCoords ,
                             eyeModel->colors  ,  eyeModel->header.numberOfColors  * sizeof(float) ,
                             0, 0 //Not Indexed..
                           );

    GLuint humanVAO;
    GLuint humanArrayBuffer;
    unsigned int humanTriangleCount  =  (unsigned int)  triModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &humanVAO,
                             &humanArrayBuffer,
                             programID  ,
                             triModel->vertices  ,  triModel->header.numberOfVertices * sizeof(float) ,
                             triModel->normal    ,  triModel->header.numberOfNormals  * sizeof(float),
                             0,0,
                             //triModel.textureCoords  ,  triModel.header.numberOfTextureCoords ,
                             triModel->colors  ,  triModel->header.numberOfColors  * sizeof(float) ,
                             0, 0 //Not Indexed..
                           );
    fprintf(stderr,"Ready to render: ");

     GLuint FramebufferName = 0;
     GLuint renderedTexture;
     GLuint renderedDepth;
     initializeFramebuffer(&FramebufferName,&renderedTexture,&renderedDepth,WIDTH,HEIGHT);

	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");

	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

	do{
        // Render to our framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        glClearColor( 0, 0.0, 0, 1 );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen

        doOGLDrawing(
                     programID,
                     MVPMatrixID,
                     eyeVAO,
                     eyeTriangleCount,
                     humanVAO,
                     humanTriangleCount,
                     WIDTH,
                     HEIGHT
                    );

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferToScreen(
                                programFrameBufferID,
                                quad_vertexbuffer,
                                //renderedDepth,
                                renderedTexture,
                                texID,
                                timeID,
                                resolutionID,
                                WIDTH,HEIGHT
                               );

		// Swap buffers
        glx3_endRedraw();
       // usleep(1);

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

	} // Check if the ESC key was pressed or the window was closed
	while( 1 );

	// Cleanup VBO and shader
	//glDeleteBuffers(1, &vertexbuffer);
	//glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &humanVAO);
	glDeleteVertexArrays(1, &eyeVAO);
}





int main(int argc,const char **argv)
{
  fprintf(stderr,"Attempting to setup a %dx%d glx3 context\n",WIDTH,HEIGHT);
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }

   #define defaultModelToLoad "makehuman.tri"
   const char * modelToLoad = defaultModelToLoad;

   //------------------------------------------------------
   struct BVH_MotionCapture mc = {0};
   //------------------------------------------------------
   struct TRI_Model indexedEyeModel={0};
   struct TRI_Model eyeModel={0};
   struct TRI_Model indexedTriModel={0};
   struct TRI_Model triModel={0};
   //------------------------------------------------------

   //------------------------------------------------------
   for (int i=0; i<argc; i++)
        {
           if (strcmp(argv[i],"--from")==0)
                    {
                        if (argc>i+1)
                            {
                                modelToLoad = argv[i+1];
                            }
                    }
        }
   //------------------------------------------------------
   if (!bvh_loadBVH("merged_neutral.bvh",&mc,1.0) ) // This is the new armature that includes the head
        {
          fprintf(stderr,"Cannot find the merged_neutral.bvh file..\n");
          return 0;
        }
   //------------------------------------------------------
   if (!loadModelTri(modelToLoad, &indexedTriModel ) )
   {
     fprintf(stderr,"Please : \n");
     fprintf(stderr,"wget http://ammar.gr/mocapnet/makehuman.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!loadModelTri("eyes.tri", &indexedEyeModel ) )
   {
     fprintf(stderr,"Please : \n");
     fprintf(stderr,"wget http://ammar.gr/mocapnet/eyes.tri\n");
     return 0;
   }
   //------------------------------------------------------

   //copyModelTri( triModelOut , triModelIn , 1 /*We also want bone data*/);
   //int applyVertexTransformation( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn )

   fillFlatModelTriFromIndexedModelTri(&triModel,&indexedTriModel);
   fillFlatModelTriFromIndexedModelTri(&eyeModel,&indexedEyeModel);

   animateTRIModelUsingBVHArmature(&indexedTriModel,&mc,0);

   doDrawing(&triModel,&eyeModel);

   stop_glx3_stuff();
 return 0;
}

