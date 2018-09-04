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

#include "../Tools/save_to_file.h"
#include "../Tools/tools.h"

#include "../ModelLoader/hardcoded_shapes.h"
#include "../ModelLoader/model_loader_tri.h"

#include "glx3.h"

#include "../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../tools/AmMatrix/quaternions.h"

#include "../Rendering/ShaderPipeline/computeShader.h"
#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"
#include "../Rendering/ShaderPipeline/uploadGeometry.h"
#include "../Rendering/ShaderPipeline/uploadTextures.h"


#include "../../../../acquisition/Acquisition.h"
#include "../../../../tools/Calibration/calibration.h"
#include "../../../../tools/Common/viewerSettings.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


#define USE_COMPUTE_SHADER 0


//Change this to change MultiRendering numbers
#define originalWIDTH 640
#define originalHEIGHT 480
#define tilesToDoX 8
#define tilesToDoY 8
#define shrinkingFactor 4
//--------------------------------------------



struct viewerSettings config={0};

unsigned int diffTextureUploaded=0;
GLuint diffTexture;
GLuint colorTexGLSLId;

unsigned int WIDTH=(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;

float lastFramerate = 60;
unsigned long lastRenderingTime = 0;
unsigned int framesRendered=0;


struct TRI_Model indexedTriModel={0};
struct TRI_Model triModel={0};

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


                 double x,
                 double y,
                 double z,
                 double roll,
                 double pitch,
                 double yaw,

                 double * projectionMatrixD,
                 double * viewportMatrixD,
                 double * viewMatrixD
                 )
{
       //Select Shader to render with
       glUseProgram(programID);                  checkOpenGLError(__FILE__, __LINE__);

       //Select Vertex Array Object To Render
       glBindVertexArray(vao);                   checkOpenGLError(__FILE__, __LINE__);

       //fprintf(stderr,"XYZRPY(%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f)\n",x,y,z,roll,pitch,yaw);


       double modelMatrixD[16];
       create4x4ModelTransformation(
                                    modelMatrixD,
                                    //Rotation Component
                                    roll,//roll
                                    pitch ,//pitch
                                    yaw ,//yaw

                                    //Translation Component (XYZ)
                                    (double) x/100,
                                    (double) y/100,
                                    (double) z/100,

                                    10.0,//scaleX,
                                    10.0,//scaleY,
                                    10.0//scaleZ
                                   );

      //-------------------------------------------------------------------
       double MVPD[16];
       float MVP[16];
       getModelViewProjectionMatrixFromMatrices(MVPD,projectionMatrixD,viewMatrixD,modelMatrixD);
       copy4x4DMatrixToF(MVP , MVPD );
       transpose4x4Matrix(MVP);
      //-------------------------------------------------------------------



		// Send our transformation to the currently bound shader, in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE/*TRANSPOSE*/, MVP);



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



int doTiledDiffDrawing(
                       int programID,
                       GLuint textureToDiff,
                       GLuint textureDiffSampler,
                       GLuint tileSizeX,
                       GLuint tileSizeY,
                       GLuint MVPMatrixID ,
                       GLuint cubeVao,
                       unsigned int cubeTriangleCount,
                       GLuint pyramidVao,
                       unsigned int pyramidTriangleCount,
                       unsigned int tilesX,
                       unsigned int tilesY
                    )
{
     // Bind our texture in Texture Unit 0
	 glActiveTexture(GL_TEXTURE1);
	 glBindTexture(GL_TEXTURE_2D, textureToDiff);

	 // Set our "renderedTexture" sampler to use Texture Unit 0
	 glUniform1i(textureDiffSampler, 1);

     //Update our tile size..
	 glUniform1i(tileSizeX,tilesX);
	 glUniform1i(tileSizeY,tilesY);


     double projectionMatrixD[16];
     double viewportMatrixD[16];
     double viewMatrixD[16];

     prepareRenderingMatrices(
                     535.423889, //fx
                     533.48468,  //fy
                     0.0,        //skew
                     (double) originalWIDTH/2,    //cx
                     (double) originalHEIGHT/2,   //cy
                     (double) originalWIDTH,      //Window Width
                     (double) originalHEIGHT,     //Window Height
                     1.0,        //Near
                     255.0,      //Far
                     projectionMatrixD,
                     viewMatrixD,
                     viewportMatrixD
                    );

     //-------------------------------------------------------------------
        double roll=0.0;//(double)  (rand()%90);
        double pitch=0.0;//(double) (rand()%90);
        double yaw=0.0;//(double)   (rand()%90);


        double quaternion[4]={-1.00,-0.13,-0.03,0.09};
        double euler[3]={0.0,0.0,0.0};
        euler2Quaternions(quaternion,euler,qXqYqZqW);

        //25.37,-87.19,2665.56,-1.00,-0.13,-0.03,0.09,0.59,0.54,0.11,-0.90,0.18,0.21,-0.43,0.65,-0.53,-1.13,0.60,0.10,0.32,-0.17,-0.09,-0.34,0.30,-0.05,0.05,0.78,0.00,-1.91,0.68,-0.07,-0.01,4.00,0.11,1.92,-0.73
        double x=-259.231f;//(double)  (1000-rand()%2000);
        double y=-54.976f;//(double) (100-rand()%200);
        double z=2699.735f;//(double)  (700+rand()%1000);
     //-------------------------------------------------------------------

  unsigned int viewportWidth = (unsigned int) WIDTH / tilesX;
  unsigned int viewportHeight = (unsigned int) HEIGHT / tilesY;

  unsigned int tx,ty;
  for (ty=0; ty<tilesY; ty++)
  {
    for (tx=0; tx<tilesX; tx++)
    {
       roll+=1.0;
       pitch+=1.5;

     glViewport(viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight );


     double newViewport[4]={viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight};
     double projectionMatrixViewportCorrected[16];
     correctProjectionMatrixForDifferentViewport(
                                                  projectionMatrixViewportCorrected,
                                                  projectionMatrixD,
                                                  viewportMatrixD,
                                                  newViewport
                                                );

     //fprintf(stderr,"glViewport(%u,%u,%u,%u)\n",viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight);
     drawObjectAT(
                  programID,
                  cubeVao,
                  MVPMatrixID,
                  cubeTriangleCount,
                  x-400,
                  y,
                  z,
                  roll,
                  pitch,
                  yaw,

                  projectionMatrixD,
                  viewportMatrixD,
                  viewMatrixD
                 );

     drawObjectAT(
                  programID,
                  pyramidVao,
                  MVPMatrixID,
                  pyramidTriangleCount,

                  25.37,-87.19,2665.56,
                  //x+1100,
                  //y,
                  //z,
                  euler[0],
                  euler[1],
                  euler[2],

                  projectionMatrixD,
                  viewportMatrixD,
                  viewMatrixD
                 );


    }
  }
  return 1;
}



void performComputeShaderOperation(struct computeShaderObject  * diffComputer)
{
    glUseProgram(diffComputer->computeShaderProgram);
    //glUniform1f(glGetUniformLocation(computeHandle, "roll"), (float)frame*0.01f);
    glDispatchCompute(512/16, 512/16, 1); // 512^2 threads in blocks of 16^2
    //checkErrors("Dispatch compute shader");
}

int uploadColorImageAsTextureFromAcquisition(
                                              GLuint programID  ,
                                              ModuleIdentifier moduleID,
                                              DeviceIdentifier devID
                                            )
{
  unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);

  return
  uploadColorImageAsTexture(
                            programID,
                            &diffTexture,
                            &diffTextureUploaded,
                            acquisitionGetColorFrame(moduleID,devID),
                            colorWidth , colorHeight , colorChannels , colorBitsperpixel
                           );
}

int uploadDepthImageAsTextureFromAcquisition(
                                              GLuint programID  ,
                                              ModuleIdentifier moduleID,
                                              DeviceIdentifier devID
                                            )
{
  unsigned int depthWidth , depthHeight , depthChannels , depthBitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&depthWidth,&depthHeight,&depthChannels,&depthBitsperpixel);

  return
  uploadDepthImageAsTexture(
                            programID,
                            &diffTexture,
                            &diffTextureUploaded,
                            acquisitionGetDepthFrame(moduleID,devID),
                            depthWidth , depthHeight , depthChannels , depthBitsperpixel
                           );
}


int doDrawing()
{
   fprintf(stderr," doDrawing \n");
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../shaders/TransformVertexShader.vertexshader", "../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../shaders/simpleDepth.vert", "../../shaders/simpleDepth.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

	struct shaderObject * textureFramebuffer = loadShader("../../shaders/virtualFramebufferTextureDiff.vert", "../../shaders/virtualFramebufferTextureDiff.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

    #if USE_COMPUTE_SHADER
    struct computeShaderObject  * diffComputer = loadComputeShader("../../shaders/virtualFramebufferTextureDiff.compute");
    if (diffComputer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }
    GLuint computeProgramID = diffComputer->computeShaderProgram;
    #endif // USE_COMPUTE_SHADER

    GLuint programID = sho->ProgramObject;
    GLuint programFrameBufferID = textureFramebuffer->ProgramObject;


 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");


	// Use our shader
	//glUseProgram(programFrameBufferID);



	// Use our shader
	glUseProgram(programID);

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    fprintf(stderr,"Ready to start pushing geometry  ");


    GLuint cubeVAO;
    GLuint cubeArrayBuffer;
    unsigned int cubeTriangleCount  =  (unsigned int )  sizeof(cubeCoords)/(3*sizeof(float));
    pushObjectToBufferData(
                             1,
                             &cubeVAO,
                             &cubeArrayBuffer,
                             programID  ,
                             cubeCoords  ,  sizeof(cubeCoords) ,
                             cubeNormals ,  sizeof(cubeNormals) ,
                             0 ,  0, //No Texture
                             cubeColors  ,  sizeof(cubeColors),
                             0, 0 //Not Indexed..
                           );


    GLuint humanVAO;
    GLuint humanArrayBuffer;
    unsigned int humanTriangleCount  =  (unsigned int)  triModel.header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &humanVAO,
                             &humanArrayBuffer,
                             programID  ,
                             triModel.vertices  ,  triModel.header.numberOfVertices * sizeof(float) ,
                             triModel.normal    ,  triModel.header.numberOfNormals  * sizeof(float),
                             0,0,
                             //triModel.textureCoords  ,  triModel.header.numberOfTextureCoords ,
                             triModel.colors  ,  triModel.header.numberOfColors  * sizeof(float) ,
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
	 GLuint textureDiffSampler = glGetUniformLocation(programFrameBufferID, "diffedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");
	 GLuint tileSizeX = glGetUniformLocation(programFrameBufferID, "tileSizeX");
	 GLuint tileSizeY = glGetUniformLocation(programFrameBufferID, "tileSizeY");

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


        //Get a new pair of frames and upload as texture..
        acquisitionSnapFrames(config.moduleID,config.devID);
        //uploadColorImageAsTexture(programFrameBufferID,config.moduleID,config.devID);
        uploadDepthImageAsTextureFromAcquisition(
                                                  programFrameBufferID,
                                                  config.moduleID,
                                                  config.devID
                                                );


        doTiledDiffDrawing(
                           programID,
                           diffTexture,
                           textureDiffSampler,
                           tileSizeX,
                           tileSizeY,
                           MVPMatrixID,
                           cubeVAO,
                           cubeTriangleCount,
                           humanVAO,
                           humanTriangleCount,
                           tilesToDoX,
                           tilesToDoY
                          );


	#if USE_COMPUTE_SHADER
        performComputeShaderOperation(diffComputer);
    #endif // USE_COMPUTE_SHADER

        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebuffer(
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
	glDeleteVertexArrays(1, &humanVAO);
	glDeleteVertexArrays(1, &cubeVAO);


	unloadShader(sho);
	unloadShader(textureFramebuffer);

	#if USE_COMPUTE_SHADER
     unloadComputeShader(diffComputer);
    #endif // USE_COMPUTE_SHADER


  return 1;
}







int main(int argc,const char **argv)
{
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }

   #define modelToLoad "../../Models/Ammar.tri"
   //#define modelToLoad "../../submodules/Assimp/Ammar_1k.tri"


   if (!loadModelTri(modelToLoad, &indexedTriModel ) )
   {
     fprintf(stderr,"please cd ../../Models/\n");
     fprintf(stderr,"and then wget http://ammar.gr/models/Ammar.tri\n");
     return 0;
   }

   fillFlatModelTriFromIndexedModelTri(&triModel,&indexedTriModel);



  /* ACQUISITION INITIALIZATION ---------------------------------------------------------------------*/
  /* ------------------------------------------------------------------------------------------------*/
  /* ------------------------------------------------------------------------------------------------*/
  acquisitionRegisterTerminationSignal(&acquisitionDefaultTerminator);
  initializeViewerSettingsFromArguments(&config,argc,argv);


  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(config.moduleID,16 /*maxDevices*/ , 0 ))
   {
       fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(config.moduleID));
       return 1;
   }

  if (!acquisitionOpenDevice(config.moduleID,config.devID,config.inputname,config.width,config.height,config.framerate) )
   {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",config.devID, config.inputname,getModuleNameFromModuleID(config.moduleID));
          return 1;
   }
/* ------------------------------------------------------------------------------------------------*/
/* ------------------------------------------------------------------------------------------------*/

   doDrawing();


   acquisitionDefaultTerminator(&config);

  stop_glx3_stuff();
 return 0;
}

