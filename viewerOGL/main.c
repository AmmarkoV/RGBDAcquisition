/** @file main.c
 *  @brief The simple Viewer that uses libAcquisition.so to view input from a module/device
 *  This should be used like ./Viewer -module TEMPLATE -from Dataset
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug No known bugs
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../tools/Calibration/calibration.h"

#include "../tools/Common/viewerSettings.h"

// Include GLEW
#include <GL/glew.h>

//GLU
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/glut.h>


struct viewerSettings config={0};


int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}

#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/System/glx3.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/shader_loader.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Tools/tools.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/ModelLoader/hardcoded_shapes.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/render_buffer.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/uploadGeometry.h"

#include "../tools/AmMatrix/matrix4x4Tools.h"
#include "../tools/AmMatrix/matrixOpenGL.h"

//OpenGL context stuff..
struct shaderObject * sho=0;
struct shaderObject * textureFramebuffer=0;

GLuint FramebufferName = 0;
GLuint renderedTexture;
GLuint renderedDepth;

GLuint quad_vertexbuffer;

GLuint MVPMatrixID;

GLuint texID;
GLuint timeID;
GLuint resolutionID;

GLuint cubeVAO;
GLuint cubeArrayBuffer;
unsigned int cubeTriangleCount;

unsigned int colorTextureUploaded=0;
GLuint colorTexture;
GLuint colorTexGLSLId;

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


       struct Matrix4x4OfFloats modelMatrix={0};
       double modelMatrixD[16];
       /*
       create4x4DModelTransformation(
                                    modelMatrixD,
                                    //Rotation Component
                                    roll,//roll
                                    pitch ,//pitch
                                    yaw ,//yaw
                                    ROTATION_ORDER_RPY,

                                    //Translation Component (XYZ)
                                    (double) x/100,
                                    (double) y/100,
                                    (double) z/100,

                                    10.0,//scaleX,
                                    10.0,//scaleY,
                                    10.0//scaleZ
                                   );*/


         create4x4FModelTransformation(
                                        &modelMatrix,
                                        //Rotation Component
                                        roll,//roll
                                        pitch ,//pitch
                                        yaw ,//yaw
                                        ROTATION_ORDER_RPY,

                                        //Translation Component (XYZ)
                                        (double) x/100,
                                        (double) y/100,
                                        (double) z/100,

                                        10.0,//scaleX,
                                        10.0,//scaleY,
                                        10.0//scaleZ
                                      );
          copy4x4FMatrixTo4x4D(modelMatrixD,modelMatrix.m);


      //-------------------------------------------------------------------
       double MVPD[16];
       float MVP[16];
       getModelViewProjectionMatrixFromMatrices(MVPD,projectionMatrixD,viewMatrixD,modelMatrixD);
       copy4x4DMatrixTo4x4F(MVP , MVPD );
       transpose4x4FMatrix(MVP);
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



int doSingleDrawing(
                   int programID,
                   GLuint MVPMatrixID ,
                   GLuint cubeVao,
                   unsigned int cubeTriangleCount,
                   unsigned int width,
                   unsigned int height
                   )
{

     double projectionMatrixD[16];
     double viewportMatrixD[16];
     double viewMatrixD[16];


     prepareRenderingMatrices(
                     535.423889, //fx
                     533.48468,  //fy
                     0.0,        //skew
                     (double) width/2,    //cx
                     (double) height/2,   //cy
                     (double) width,      //Window Width
                     (double) height,     //Window Height
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

        double x=-259.231f;//(double)  (1000-rand()%2000);
        double y=-54.976f;//(double) (100-rand()%200);
        double z=2699.735f;//(double)  (700+rand()%1000);
     //-------------------------------------------------------------------
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

 return 1;
}



int uploadColorImageAsTexture(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);

  glUseProgram(textureFramebuffer->ProgramObject);

    if (colorTextureUploaded)
     {
       glDeleteTextures(1,&colorTexture);
       colorTextureUploaded=0;
     }



    glEnable(GL_TEXTURE_2D);
    glGenTextures(1,&colorTexture);
    colorTextureUploaded=1;
    glBindTexture(GL_TEXTURE_2D,colorTexture);

      /* LOADING TEXTURE --WITHOUT-- MIPMAPING - IT IS LOADED RAW*/
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);                       //GL_RGB
      glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    colorWidth ,
                    colorHeight,
                    0,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    (const GLvoid *) acquisitionGetColorFrame(moduleID,devID)
                  );

    glFlush();
    return 1;
}


int acquisitionCreateDisplay(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);

  if (!start_glx3_stuff(colorWidth,colorHeight,1,0,0)) { fprintf(stderr,"Could not initialize"); return 1;}


  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 0;
   }

    sho = loadShader("../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/shaders/simple.vert",
                     "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/shaders/simple.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

	textureFramebuffer = loadShader("../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/shaders/virtualFramebuffer.vert",
                                    "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/shaders/virtualFramebufferTextureInput.frag"
                                    //"../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/shaders/virtualFramebufferFlow.frag"
                                    );
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }







 	// Get a handle for our "MVP" uniform
	MVPMatrixID = glGetUniformLocation(sho->ProgramObject, "MVP");


	// Use our shader
	glUseProgram(sho->ProgramObject);

	// Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);

	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

    fprintf(stderr,"Ready to start pushing geometry  ");


    cubeTriangleCount  =  (unsigned int )  sizeof(cubeCoords)/(3*sizeof(float));
    pushObjectToBufferData(
                             1,
                             &cubeVAO,
                             &cubeArrayBuffer,
                             sho->ProgramObject  ,
                             cubeCoords  ,  sizeof(cubeCoords) ,
                             cubeNormals ,  sizeof(cubeNormals) ,
                             0 ,  0, //No Texture
                             cubeColors  ,  sizeof(cubeColors),
                             0, 0 //Not Indexed..
                           );







     initializeFramebuffer(&FramebufferName,&renderedTexture,&renderedDepth,config->width,config->height);

	 glGenBuffers(1, &quad_vertexbuffer);
	 glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	 glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	 // Create and compile our GLSL program from the shaders
	 GLuint texID = glGetUniformLocation(textureFramebuffer->ProgramObject, "renderedTexture");
	 // Create and compile our GLSL program from the shaders
	 GLuint timeID = glGetUniformLocation(textureFramebuffer->ProgramObject, "iTime");

	 GLuint resolutionID = glGetUniformLocation(textureFramebuffer->ProgramObject, "iResolution");

	 colorTexGLSLId =  glGetUniformLocation(textureFramebuffer->ProgramObject, "iResolution");

  return 1;
}

int acquisitionDisplayFrames(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int framerate)
{

    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glViewport(0,0,config->width,config->height); // Render on the whole framebuffer, complete from the lower left corner to the upper right


        glClearColor( 0, 0.0, 0, 1 );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 		// Clear the screen




        uploadColorImageAsTexture(moduleID,devID);


/*
       doSingleDrawing(
                              sho->ProgramObject,
                              MVPMatrixID,
                              cubeVAO,
                              cubeTriangleCount,
                              config->width,
                              config->height
                             );
*/

       //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebufferFromTexture(
                        colorTexture,
                        textureFramebuffer->ProgramObject,
                        quad_vertexbuffer,
                        //renderedDepth,
                        renderedTexture,
                        texID,
                        timeID,
                        resolutionID,
                        config->width,config->height
                       );


		// Swap buffers

  glx3_endRedraw();
 return 1;
}


int acquisitionStopDisplayingFrames(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID)
{
     stop_glx3_stuff();
   	 return 1;
}



void closeEverything(struct viewerSettings * config)
{
 fprintf(stderr,"Gracefully closing everything .. ");

 acquisitionStopDisplayingFrames(config,config->moduleID,config->devID);
 /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
 acquisitionCloseDevice(config->moduleID,config->devID);

 if (config->devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionCloseDevice(
                                  config->moduleID,
                                  config->devID2
                                );
        }

 acquisitionStopModule(config->moduleID);

 fprintf(stderr,"Done\n");
 exit(0);
}






int main(int argc,const char *argv[])
{
  acquisitionRegisterTerminationSignal(&closeEverything);

  initializeViewerSettingsFromArguments(&config,argc,argv);


  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(config.moduleID,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(config.moduleID));
       return 1;
   }

  if (config.drawColor==0) { acquisitionDisableStream(config.moduleID,config.devID,0); }
  if (config.drawDepth==0) { acquisitionDisableStream(config.moduleID,config.devID,1); }


  //We want to check if deviceID we requested is a logical value , or we dont have that many devices!
  unsigned int maxDevID=acquisitionGetModuleDevices(config.moduleID);
  if ( (maxDevID==0) && (!acquisitionMayBeVirtualDevice(config.moduleID,config.devID,config.inputname)) ) { fprintf(stderr,"No devices availiable , and we didn't request a virtual device\n");  return 1; }
  if ( maxDevID < config.devID ) { fprintf(stderr,"Device Requested ( %u ) is out of range ( only %u available devices ) \n",config.devID,maxDevID);  return 1; }
  //If we are past here we are good to go..!


   if ( config.calibrationSet )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",config.calib.farPlane,config.calib.nearPlane);
    acquisitionSetColorCalibration(config.moduleID,config.devID,&config.calib);
    acquisitionSetDepthCalibration(config.moduleID,config.devID,&config.calib);
   }

  char * devName = config.inputname;
  if (strlen(config.inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
      if (config.seekFrame!=0)
      {
          acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
      }
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        if (!acquisitionOpenDevice(config.moduleID,config.devID,devName,config.width,config.height,config.framerate) )
        {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",config.devID,devName,getModuleNameFromModuleID(config.moduleID));
          return 1;
        }
        if ( strstr(config.inputname,"noRegistration")!=0 )         {  } else
        if ( strstr(config.inputname,"rgbToDepthRegistration")!=0 ) { acquisitionMapRGBToDepth(config.moduleID,config.devID); } else
                                                                    { acquisitionMapDepthToRGB(config.moduleID,config.devID); }
        fprintf(stderr,"Done with Mapping Depth/RGB \n");

        if (config.devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionOpenDevice(config.moduleID,config.devID2,devName,config.width,config.height,config.framerate);
        }


     sprintf(config.RGBwindowName,"RGBDAcquisition RGB - Module %u Device %u",config.moduleID,config.devID);
     sprintf(config.DepthwindowName,"RGBDAcquisition Depth - Module %u Device %u",config.moduleID,config.devID);

      if (config.seekFrame!=0)
      {
          acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
      }



   acquisitionCreateDisplay(
                             &config,
                              config.moduleID,
                              config.devID
                           );


   while ( (!config.stop) && ( (config.maxFramesToGrab==0)||(config.frameNum<config.maxFramesToGrab) ) )
    {
        if (config.verbose)
        {
           fprintf(stderr,"Frame Number is : %u\n",config.frameNum);
        }
        acquisitionStartTimer(0);

        acquisitionSnapFrames(config.moduleID,config.devID);

        acquisitionDisplayFrames(&config,config.moduleID,config.devID,config.framerate);

       if (config.devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionSnapFrames(config.moduleID,config.devID2);
          acquisitionDisplayFrames(&config,config.moduleID,config.devID2,config.framerate);
        }

        acquisitionStopTimer(0);
        if (config.frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
        ++config.frameNum;

       if ( config.waitKeyToStart>0 )
       {
         --config.waitKeyToStart;
       }

        if (config.loopFrame!=0)
        {
          //fprintf(stderr,"%u%%(%u+%u)==%u\n",frameNum,loopFrame,seekFrame,frameNum%(loopFrame+seekFrame));
          if ( config.frameNum%(config.loopFrame)==0)
          {
            fprintf(stderr,"Looping Dataset , we reached frame %u ( %u ) , going back to %u\n",config.frameNum,config.loopFrame,config.seekFrame);
            acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
          }
        }


      if (config.executeEveryLoopPayload)
      {
         int i=system(config.executeEveryLoop);
         if (i!=0) { fprintf(stderr,"Could not execute payload\n"); }
      }

      if (config.delay>0)
      {
         usleep(config.delay);
      }

    }

    fprintf(stderr,"Done viewing %u frames! \n",config.frameNum);

    closeEverything(&config);

    return 0;
}

