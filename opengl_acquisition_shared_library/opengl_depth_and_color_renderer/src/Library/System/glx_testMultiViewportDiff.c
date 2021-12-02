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
#include <math.h>

#include "../Tools/save_to_file.h"
#include "../Tools/tools.h"

#include "../ModelLoader/hardcoded_shapes.h"
#include "../ModelLoader/model_loader_tri.h"

#include "glx3.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../../tools/AmMatrix/quaternions.h"

#include "../Rendering/ShaderPipeline/computeShader.h"
#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"
#include "../Rendering/ShaderPipeline/uploadGeometry.h"
#include "../Rendering/ShaderPipeline/uploadTextures.h"
#include "../Rendering/downloadFromRenderer.h"


#include "../../../../../acquisition/Acquisition.h"
#include "../../../../../tools/Calibration/calibration.h"
#include "../../../../../tools/Common/viewerSettings.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


#define USE_COMPUTE_SHADER 0


//Change this to change MultiRendering numbers
#define originalWIDTH 640
#define originalHEIGHT 480
#define tilesToDoX 8
#define tilesToDoY 8
#define shrinkingFactor 4 //4
#define drawShrinkingFactor 4
//--------------------------------------------

struct viewerSettings config={0};

unsigned int diffTextureUploaded=0;
GLuint diffTexture;
GLuint colorTexGLSLId;


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




float calculateScoresForTile(unsigned char * data , unsigned int x1, unsigned int y1,unsigned int width, unsigned int height,unsigned int globalWidth,unsigned int globalHeight)
{
  unsigned int tX=0;
  unsigned int tY=0;

  unsigned int sumDistance=0;
  unsigned int sumCorrect=0;
  unsigned int sumRendered=0;

  unsigned char * ptr;
  //fprintf(stderr,"(%u,%u)=>(%u,%u)/(%u,%u) \n",x1,y1,x1+width,y1+height,globalWidth,globalHeight);
  for (tY=y1; tY<y1+height; tY++)
  {
   for (tX=x1; tX<x1+width; tX++)
   {
     ptr = data+(tY*globalWidth*3+tX*3);
     //fprintf(stderr,"(");
     sumDistance += *ptr;                     /*fprintf(stderr,"%u ",(unsigned char) *ptr);*/ ptr++;
     sumCorrect += (unsigned int) ((*ptr)>0); /*fprintf(stderr,"%u ",(unsigned char) *ptr);*/ ptr++;
     sumRendered += (unsigned int) ((*ptr)>0); /*fprintf(stderr,"%u",(unsigned char) *ptr);*/ ptr++;
     //fprintf(stderr,")");
   }
  }

  //fprintf(stderr,"Score(%u,%u)=%u %u %u\n",tX,tY,sumDistance,sumCorrect,sumRendered);
  if (sumRendered == 0 ) { sumRendered=1; }
  //float returnValue = (float) sumDistance * ( (float) sumCorrect/sumRendered );
  if (sumCorrect == 0 ) { sumCorrect=1; }
  float returnValue = (float) sumDistance / sumCorrect;
  return returnValue;
}


int calculateScores(unsigned char * data , unsigned int tilesX, unsigned int tilesY,unsigned int width, unsigned int height)
{
  unsigned int tX=0;
  unsigned int tY=0;
  unsigned int tWidth=(unsigned int) width/tilesX;
  unsigned int tHeight=(unsigned int) height/tilesY;


  unsigned int bestTileX=666,bestTileY=666;
  float currentScore,bestScore=100000000;

  for (tY=0; tY<tilesY; tY++)
  {
   for (tX=0; tX<tilesX; tX++)
   {
       //fprintf(stderr,"calculateScoresForTile(%u,%u) - ",tX,tY);
       currentScore = calculateScoresForTile(data,tX*tWidth,tY*tHeight,tWidth,tHeight,width,height);

       if (currentScore<=bestScore)
       {
          bestTileX=tX;
          bestTileY=tY;
          bestScore=currentScore;
       }
   }
  }

  fprintf(stderr,"Best Tile is %u,%u with %0.2f score \n",bestTileX,bestTileY,currentScore);

  return 1;
}








int drawObjectAT(GLuint programID,
                 GLuint vao,
                 GLuint MVPMatrixID,
                 GLuint MVMatrixID ,
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
                                    (float) x/100,
                                    (float) y/100,
                                    (float) z/100,

                                    10.0,//scaleX,
                                    10.0,//scaleY,
                                    10.0//scaleZ
                                   );

      //-------------------------------------------------------------------
       struct Matrix4x4OfFloats MVP;
       getModelViewProjectionMatrixFromMatrices(&MVP,projectionMatrix,viewMatrix,&modelMatrix);
       transpose4x4FMatrix(MVP.m);


       struct Matrix4x4OfFloats MV;
       multiplyTwo4x4FMatricesS(&MV,viewMatrix,&modelMatrix);
       transpose4x4FMatrix(MV.m);
      //-------------------------------------------------------------------

      // Send our transformation to the currently bound shader, in the "MVP" uniform
      glUniformMatrix4fv(MVPMatrixID, 1, GL_FALSE/*TRANSPOSE*/, MVP.m);
      glUniformMatrix4fv(MVMatrixID, 1, GL_FALSE/*TRANSPOSE*/, MV.m);



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
                       GLuint FramebufferName,
                       int programID,
                       int programFrameBufferID,
                       GLuint textureToDiff,
                       GLuint textureDiffSampler,
                       GLuint tileSizeX,
                       GLuint tileSizeY,
                       GLuint MVPMatrixID ,
                       GLuint MVMatrixID ,
                       GLuint cubeVao,
                       unsigned int cubeTriangleCount,
                       GLuint pyramidVao,
                       unsigned int pyramidTriangleCount,
                       unsigned int tilesX,
                       unsigned int tilesY,
                       unsigned int WIDTH,unsigned int HEIGHT
                    )
{
     glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	 glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

     glClearColor( 0, 0.0, 0, 1 );
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen


     glUseProgram(programFrameBufferID);
     // Bind our texture in Texture Unit 0
	 glActiveTexture(GL_TEXTURE1);
	 glBindTexture(GL_TEXTURE_2D, textureToDiff);
	 // Set our "renderedTexture" sampler to use Texture Unit 0
	 glUniform1i(textureDiffSampler, 1);

     //Update our tile size..
	 glUniform1i(tileSizeX,tilesX);
	 glUniform1i(tileSizeY,tilesY);


     struct Matrix4x4OfFloats  projectionMatrix;
     struct Matrix4x4OfFloats  viewportMatrix;
     struct Matrix4x4OfFloats  viewMatrix;

     prepareRenderingMatrices(
                              535.423889, //fx
                              533.48468,  //fy
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
        float roll=0.0;//(float)  (rand()%90);
        float pitch=0.0;//(float) (rand()%90);
//        float yaw=0.0;//(float)   (rand()%90);


        float quaternion[4]={-1.00,-0.13,-0.03,0.09};
        float euler[3]={0.0,0.0,0.0};
        euler2Quaternions(quaternion,euler,qXqYqZqW);

        //25.37,-87.19,2665.56,-1.00,-0.13,-0.03,0.09,0.59,0.54,0.11,-0.90,0.18,0.21,-0.43,0.65,-0.53,-1.13,0.60,0.10,0.32,-0.17,-0.09,-0.34,0.30,-0.05,0.05,0.78,0.00,-1.91,0.68,-0.07,-0.01,4.00,0.11,1.92,-0.73
//        float x=-259.231f;//(float)  (1000-rand()%2000);
//        float y=-54.976f;//(float) (100-rand()%200);
//        float z=2699.735f;//(float)  (700+rand()%1000);
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


     float newViewport[4]={viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight};
     float projectionMatrixViewportCorrected[16];
     correctProjectionMatrixForDifferentViewport(
                                                  projectionMatrixViewportCorrected,
                                                  projectionMatrix.m,
                                                  viewportMatrix.m,
                                                  newViewport
                                                );

     //fprintf(stderr,"glViewport(%u,%u,%u,%u)\n",viewportWidth*tx, viewportHeight*ty, viewportWidth , viewportHeight);
     /*
     drawObjectAT(
                  programID,
                  cubeVao,
                  MVPMatrixID,
                  MVMatrixID,
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
                 );*/

     unsigned int distance = 500;
     unsigned int distanceMul2 = distance * 2;
     float xDivergance = (float) distance - rand()%distanceMul2;
     float yDivergance = (float) distance - rand()%distanceMul2;
     float zDivergance = (float) distance - rand()%distanceMul2;

     drawObjectAT(
                  programID,
                  pyramidVao,
                  MVPMatrixID,
                  MVMatrixID,
                  pyramidTriangleCount,

                  25.37 + xDivergance,
                 -87.19 + yDivergance,
                  1850.0+ zDivergance,
                  euler[0],
                  euler[1],
                  euler[2],

                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix
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


int doDrawing( unsigned int WIDTH, unsigned int HEIGHT ,
               unsigned int drawWIDTH , unsigned int drawHEIGHT )
{
   fprintf(stderr," doDrawing \n");
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../shaders/TransformVertexShader.vertexshader", "../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../../shaders/simpleDepth.vert", "../../../shaders/simpleDepth.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

	struct shaderObject * textureFramebuffer = loadShader("../../../shaders/virtualFramebufferTextureDiff.vert", "../../../shaders/virtualFramebufferTextureDiff.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }


	struct shaderObject * finalFramebuffer = loadShader("../../../shaders/virtualFramebufferTextureInputNoFlip.vert", "../../../shaders/virtualFramebufferTextureInputNoFlip.frag");
    if (finalFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }


    #if USE_COMPUTE_SHADER
    struct computeShaderObject  * diffComputer = loadComputeShader("../../shaders/virtualFramebufferTextureDiff.compute");
    if (diffComputer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }
    GLuint computeProgramID = diffComputer->computeShaderProgram;
    #endif // USE_COMPUTE_SHADER

    GLuint programID = sho->ProgramObject;
    GLuint programFrameBufferID = textureFramebuffer->ProgramObject;


 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
	GLuint MVMatrixID  = glGetUniformLocation(programID, "MV");

	// Use our shader
	glUseProgram(programID);

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	//--------------------------------------------------------
    //fprintf(stderr,"Ready to start pushing geometry  ");


    GLuint cubeVAO;
    GLuint cubeArrayBuffer;
    GLuint cubeElementBuffer;
    unsigned int cubeTriangleCount  =  (unsigned int )  sizeof(cubeCoords)/(3*sizeof(float));
    pushObjectToBufferData(
                             1,
                             &cubeVAO,
                             &cubeArrayBuffer,
                             &cubeElementBuffer,
                             programID  ,
                             cubeCoords  ,  sizeof(cubeCoords) ,
                             cubeNormals ,  sizeof(cubeNormals) ,
                             0 ,  0, //No Texture
                             cubeColors  ,  sizeof(cubeColors),
                             0, 0 //Not Indexed..
                           );


    GLuint humanVAO;
    GLuint humanArrayBuffer;
    GLuint humanElementBuffer;
    unsigned int humanTriangleCount  =  (unsigned int)  triModel.header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &humanVAO,
                             &humanArrayBuffer,
                             &humanElementBuffer,
                             programID  ,
                             triModel.vertices  ,  triModel.header.numberOfVertices * sizeof(float) ,
                             triModel.normal    ,  triModel.header.numberOfNormals  * sizeof(float),
                             0,0,
                             //triModel.textureCoords  ,  triModel.header.numberOfTextureCoords ,
                             triModel.colors  ,  triModel.header.numberOfColors  * sizeof(float) ,
                             0, 0 //Not Indexed..
                           );
//    fprintf(stderr,"Ready to render: ");


     GLuint FramebufferName2; // This framebuffer is the second framebuffer to use and will have renderedTexture2 assigned to it
     GLuint renderedTexture2; // This texture will hold our result
     fprintf(stderr,"Initializing framebuffers..\n");
     initializeFramebuffer(&FramebufferName2,&renderedTexture2,0/*depth*/,WIDTH,HEIGHT);


     GLuint FramebufferName;
     GLuint renderedTexture;
     //GLuint renderedDepth;
     initializeFramebuffer(&FramebufferName,&renderedTexture,0 /*&renderedDepth*/,WIDTH,HEIGHT);
     fprintf(stderr,"Done..\n");

	 GLuint quad_vertexbuffer;
	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint renderedTextureGLSLName = glGetUniformLocation(programFrameBufferID, "renderedTexture");
	 GLuint textureDiffSampler = glGetUniformLocation(programFrameBufferID, "diffedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(programFrameBufferID, "iTime");
	 GLuint tileSizeX = glGetUniformLocation(programFrameBufferID, "tileSizeX");
	 GLuint tileSizeY = glGetUniformLocation(programFrameBufferID, "tileSizeY");

	 GLuint resolutionID = glGetUniformLocation(programFrameBufferID, "iResolution");

     unsigned char * retrievedData = (unsigned char*) malloc(sizeof(unsigned char) * drawWIDTH * drawHEIGHT * 3 );
     if (retrievedData==0) { return 0;}

	do
     {
        // Render to our framebuffer
		//glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
		//glViewport(0,0,WIDTH,HEIGHT); // Render on the whole framebuffer, complete from the lower left corner to the upper right

       //-----------------------------------------------
        if (framesRendered%10==0) { fprintf(stderr,"\r%0.2f FPS                                         \r", lastFramerate ); }
       //-----------------------------------------------

        //Get a new pair of frames and upload as texture..
        acquisitionSnapFrames(config.moduleID,config.devID);
        //uploadColorImageAsTexture(programFrameBufferID,config.moduleID,config.devID);
        uploadDepthImageAsTextureFromAcquisition(
                                                  programFrameBufferID,
                                                  config.moduleID,
                                                  config.devID
                                                );

        //We first want to draw using tiles to framebuffer ( FramebufferName )
        doTiledDiffDrawing(
                           FramebufferName,
                           programID,
                           programFrameBufferID,
                           diffTexture,
                           textureDiffSampler,
                           tileSizeX,
                           tileSizeY,
                           MVPMatrixID,
                           MVMatrixID,
                           cubeVAO,
                           cubeTriangleCount,
                           humanVAO,
                           humanTriangleCount,
                           tilesToDoX,
                           tilesToDoY,
                           WIDTH,HEIGHT
                          );


	#if USE_COMPUTE_SHADER
        performComputeShaderOperation(diffComputer);
    #endif // USE_COMPUTE_SHADER


    #define DO_SECOND_STAGE 1

    //----------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------
	// Use our shader
	#if DO_SECOND_STAGE
     //glFinish();

	 glUseProgram(finalFramebuffer->ProgramObject);
/*
	 //------------------------
	 GLuint renderedTextureGLSLName = glGetUniformLocation(finalFramebuffer->ProgramObject, "renderedTexture"); //We want two buffers for this thing..
     // Bind our texture in Texture Unit 0
	 glActiveTexture(GL_TEXTURE1);
	 glBindTexture(GL_TEXTURE_2D, renderedTexture);
	 // Set our "renderedTexture" sampler to use Texture Unit 0
     glUniform1i(renderedTextureGLSLName, 1);
     //------------------------
*/

	 GLuint renderedTexture2GLSLName  = glGetUniformLocation(finalFramebuffer->ProgramObject, "renderedTexture2");
	 GLuint timeID2 = glGetUniformLocation(finalFramebuffer->ProgramObject, "iTime");
	 GLuint resolutionID2 = glGetUniformLocation(finalFramebuffer->ProgramObject, "iResolution");

    //----------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------


    //We first want to draw using the textureFramebuffer and store output to a new framebuffer
    drawFramebufferTexToTex(
                            FramebufferName2,    //Target Framebuffer
                            programFrameBufferID,//GLSL Shader to use
                            quad_vertexbuffer,   //Rectangle that covers all the screen
                            renderedTexture,     // Texture to Draw
                            renderedTextureGLSLName,               // Texture to Draw GLSL uniform location
                            // Control variables that get auto populated
                            timeID,
                            resolutionID,
                            //Resolution of our Rendered Texture
                            WIDTH,HEIGHT
                           );
      checkOpenGLError(__FILE__, __LINE__);

     //We generate mipmaps to render to a really small
     //Texture without destroying quality
     glFlush();
     glBindTexture(GL_TEXTURE_2D, renderedTexture2);
     glGenerateMipmap(GL_TEXTURE_2D);
     glFlush();


	 //usleep(100000);
	 //glFinish();
     //We have accumulated all data on the framebuffer and will now draw it back..
    drawFramebufferToScreen(
                            finalFramebuffer->ProgramObject, //GLSL Shader to use
                            quad_vertexbuffer,//Rectangle that covers all the screen
                            renderedTexture2, // Texture to Draw
                            renderedTexture2GLSLName,   // Texture to Draw GLSL uniform location
                            // Control variables that get auto populated
                            timeID2,
                            resolutionID2,
                            //Resolution of our Rendered Texture
                            drawWIDTH,drawHEIGHT
                           );

     if ( downloadOpenGLColor(retrievedData,0,0,drawWIDTH,drawHEIGHT) )
     {
         calculateScores(retrievedData,tilesToDoX,tilesToDoY,drawWIDTH,drawHEIGHT);
         //saveRawImageToFileOGLR("retreivedData.pnm",retrievedData,drawWIDTH,drawHEIGHT,3,8);
     }

      checkOpenGLError(__FILE__, __LINE__);
      glFlush();
	 #else
          drawFramebufferToScreen(
                                  programFrameBufferID,
                                  quad_vertexbuffer,
                                  renderedTexture,
                                  texID,
                                  timeID,
                                  resolutionID,
                                  drawWIDTH,drawHEIGHT
                                 );
	 #endif // DO_SECOND_STAGE

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
	} // Check if the ESC key was pressed or the window was closed
	while( 1 );

	// Cleanup VBO and shader
	glDeleteVertexArrays(1, &humanVAO);
	glDeleteVertexArrays(1, &cubeVAO);


	unloadShader(sho);
	unloadShader(textureFramebuffer);
	unloadShader(finalFramebuffer);

	#if USE_COMPUTE_SHADER
     unloadComputeShader(diffComputer);
    #endif // USE_COMPUTE_SHADER

  return 1;
}





int main(int argc,const char **argv)
{
  //testPackUnpack();
  unsigned int WIDTH=(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
  unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;

  unsigned int drawWIDTH = WIDTH/drawShrinkingFactor;
  unsigned int drawHEIGHT = HEIGHT/drawShrinkingFactor;


  start_glx3_stuff(drawWIDTH,drawHEIGHT,1,argc,argv);

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }

   #define modelToLoad "../../../Models/Ammar.tri"
   //#define modelToLoad "../../../submodules/Assimp/Ammar_1k.tri"


   if (!loadModelTri(modelToLoad, &indexedTriModel ) )
   {
     fprintf(stderr,"please cd ../../../Models/\n");
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

  doDrawing(WIDTH,HEIGHT,drawWIDTH,drawHEIGHT);


  acquisitionDefaultTerminator(&config);

  stop_glx3_stuff();
 return 0;
}

