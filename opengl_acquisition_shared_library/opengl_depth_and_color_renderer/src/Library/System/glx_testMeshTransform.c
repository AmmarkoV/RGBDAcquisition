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

 struct pose6D
 {
     float x,y,z;
     float roll,pitch,yaw;

     char usePoseMatrixDirectly;
     struct Matrix4x4OfFloats m;
 };

unsigned int WIDTH =(unsigned int) (tilesToDoX*originalWIDTH)/shrinkingFactor;
unsigned int HEIGHT=(unsigned int) (tilesToDoY*originalHEIGHT)/shrinkingFactor;

float lastFramerate = 60;
unsigned long lastRenderingTime = 0;
unsigned int framesRendered = 0;


int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}


/*
static const char * OpenCOLLADANames[]=
{
"Hips", // 0 "hip",
"Hips", // 3  "hip",
"Spine", // 6  "abdomen",
 "Spine1", // 9  "chest",
 "Neck", // 12  "neck",
 "Neck1", // 15  "neck1",
 "Head", // 18  "head",
 "-", // 21  "__jaw",
 "jaw", // 24  "jaw",
 "special04", // 27 "special04",
 "oris02", // 30 "oris02",
 "oris01", // 33 "oris01",
 "oris06_L", // 36  "oris06.l",
 "oris07_L", // 39  "oris07.l",
 "oris06_R", // 42 "oris06.r",
 "oris07_R", // 45  "oris07.r",
 "tongue00", // 48 "tongue00",
 "tongue01", // 51 "tongue01",
 "tongue02", // 54 "tongue02",
 "tongue03", // 57 "tongue03",
 "-", // 60  "__tongue04",
 "tongue04", // 63 "tongue04",
 "tongue07_L", // 66 "tongue07.l",
 "tongue07_R", // 69 "tongue07.r",
 "tongue06_L", // 72 "tongue06.l",
 "tongue06_R", // 75 "tongue06.r",
 "tongue05_L", // 78 "tongue05.l",
 "tongue05_R", // 81 "tongue05.r",
 "-", // 84 "__levator02.l",
 "levator02_L", // 87 "levator02.l",
 "levator03_L", // 90 "levator03.l",
 "levator04_L", // 93 "levator04.l",
 "levator05_L", // 96 "levator05.l",
 "-", // 99 "__levator02.r",
 "levator02_R", // 102 "levator02.r",
 "levator03_R", // 105 "levator03.r",
 "levator04_R", // 108 "levator04.r",
 "levator05_R", // 111 "levator05.r",
 "-", // 114 "__special01",
 "special01", // 117 "special01",
 "oris04_L", // 120   "oris04.l",
 "oris03_L", // 123 "oris03.l",
 "oris04_R", // 126 "oris04.r",
 "oris03_R", // 129 "oris03.r",
 "oris06", // 132 "oris06",
 "oris05", // 135 "oris05",
 "-", // 138 "__special03",
 "special03", // 141 "special03",
 "-", // 144 "__levator06.l",
 "levator06_L", // 147 "levator06.l",
 "-", // 150 "__levator06.r",
 "levator06_R", // 153 "levator06.r",
 "special06_L", // 156 "special06.l",
 "special05_L", // 159 "special05.l",
 "eye_L", // 162 "eye.l",
 "orbicularis03_L", // 165  "orbicularis03.l",
 "orbicularis04_L", // 168 "orbicularis04.l",
 "special06_R", // 171 "special06.r",
 "special05_R", // 174 "special05.r",
 "eye_R", // 177 "eye.r",
 "orbicularis03_R", // 180 "orbicularis03.r",
 "orbicularis04_R", // 183 "orbicularis04.r",
 "-", // 186 "__temporalis01.l",
 "temporalis01_L", // 189 "temporalis01.l",
 "oculi02_L", // 192 "oculi02.l",
 "oculi01_L", // 195 "oculi01.l",
 "__temporalis01_R", // 198 "__temporalis01.r",
 "temporalis01_R", // 201 "temporalis01.r",
 "oculi02_R", // 204 "oculi02.r",
 "oculi01_R", // 207 "oculi01.r",
 "__temporalis02_L", // 210 "__temporalis02.l",
 "temporalis02_L", // 213 "temporalis02.l",
 "risorius02_L", // 216 "risorius02.l",
 "risorius03_L", // 219 "risorius03.l",
 "__temporalis02_R", // 222 "__temporalis02.r",
 "temporalis02_R", // 225 "temporalis02.r",
 "risorius02_R", // 228 "risorius02.r",
 "risorius03_R", // 231 "risorius03.r",
 "RightShoulder", // 234 "rCollar",
 "RightArm", // 237 "rShldr",
 "RightForeArm", // 240  "rForeArm",
 "RightHand", // 243"rHand",
 "metacarpal1_R", // 246 "metacarpal1.r",
 "finger2-1_R", // 249 "finger2-1.r",
 "finger2-2_R", // 252  "finger2-2.r",
 "finger2-3_R", // 255  "finger2-3.r",
 "metacarpal2_R", // 258 "metacarpal2.r",
 "finger3-1_R", // 261 "finger3-1.r",
 "finger3-2_R", // 264 "finger3-2.r",
 "finger3-3_R", // 267 "finger3-3.r",
 "__metacarpal3_R", // 270  "__metacarpal3.r",
 "metacarpal3_R", // 273  "metacarpal3.r",
 "finger4-1_R", // 276  "finger4-1.r",
 "finger4-2_R", // 279 "finger4-2.r",
 "finger4-3_R", // 282  "finger4-3.r",
 "__metacarpal4_R", // 285  "__metacarpal4.r",
 "metacarpal4_R", // 288 "metacarpal4.r",
 "finger5-1_R", // 291 "finger5-1.r",
 "finger5-2_R", // 294 "finger5-2.r",
 "finger5-3_R", // 297  "finger5-3.r",
 "__rthumb", // 300 "__rthumb",
 "RThumb", // 303 "rthumb",
 "finger1-2_R", // 306 "finger1-2.r",
 "finger1-3_R", // 309 "finger1-3.r",
 "LeftShoulder", // 312 "lCollar",
 "LeftArm", // 315 "lShldr",
 "LeftForeArm", // 318 "lForeArm",
 "LeftHand", // 321 "lHand",
 "metacarpal1_L", // 324  "metacarpal1.l",
 "finger2-1_L", // 327 "finger2-1.l",
 "finger2-2_L", // 330 "finger2-2.l",
 "finger2-3_L", // 333 "finger2-3.l",
 "metacarpal2_L", // 336 "metacarpal2.l",
 "finger3-1_L", // 339 "finger3-1.l",
 "finger3-2_L", // 342 "finger3-2.l",
 "finger3-3_L", // 345 "finger3-3.l",
 "__metacarpal3_L", // 348 "__metacarpal3.l",
 "metacarpal3_L", // 351 "metacarpal3.l",
 "finger4-1_L", // 354 "finger4-1.l",
 "finger4-2_L", // 357 "finger4-2.l",
 "finger4-3_L", // 360 "finger4-3.l",
 "__metacarpal4_L", // 363 "__metacarpal4.l",
 "metacarpal4_L", // 366 "metacarpal4.l",
 "finger5-1_L", // 369 "finger5-1.l",
 "finger5-2_L", // 372 "finger5-2.l",
 "finger5-3_L", // 375 "finger5-3.l",
 "__lthumb", // 378 "__lthumb",
 "LThumb", // 381 "lthumb",
 "finger1-2_L", // 384 "finger1-2.l",
 "finger1-3_L", // 387 "finger1-3.l",
 "RHipJoint", // 390 "rButtock",
 "RightUpLeg", // 393  "rThigh",
 "RightLeg", // 396 "rShin",
 "RightFoot", // 399 "rFoot",
 "_toe1-1_R", // 402 "toe1-1.R",  <- This does not currently exist in the makehuman model
 "_toe1-2_R", // 405 "toe1-2.R", <- This does not currently exist in the makehuman model
 "_toe2-1_R", // 408 "toe2-1.R", <- This does not currently exist in the makehuman model
 "_toe2-2_R", // 411 "toe2-2.R", <- This does not currently exist in the makehuman model
 "_toe2-3_R", // 414 "toe2-3.R", <- This does not currently exist in the makehuman model
 "_toe3-1_R", // 417 "toe3-1.R", <- This does not currently exist in the makehuman model
 "_toe3-2_R", // 420 "toe3-2.R", <- This does not currently exist in the makehuman model
 "_toe3-3_R", // 423 "toe3-3.R", <- This does not currently exist in the makehuman model
 "_toe4-1_R", // 426 "toe4-1.R", <- This does not currently exist in the makehuman model
 "_toe4-2_R", // 429 "toe4-2.R", <- This does not currently exist in the makehuman model
 "_toe4-3_R", // 432 "toe4-3.R", <- This does not currently exist in the makehuman model
 "_toe5-1_R", // 435 "toe5-1.R", <- This does not currently exist in the makehuman model
 "_toe5-2_R", // 438 "toe5-2.R", <- This does not currently exist in the makehuman model
 "_toe5-3_R", // 441 "toe5-3.R", <- This does not currently exist in the makehuman model
 "LHipJoint", // 444 "lButtock",
 "LeftUpLeg", // 447  "lThigh",
 "LeftLeg", // 450 "lShin",
 "LeftFoot", // 453 "lFoot",
 "_toe1-1_L", // 456 "toe1-1.L", <- This does not currently exist in the makehuman model
 "_toe1-2_L", // 459 "toe1-2.L",  <- This does not currently exist in the makehuman model
 "_toe2-1_L", // 462 "toe2-1.L", <- This does not currently exist in the makehuman model
 "_toe2-2_L", // 465 "toe2-2.L", <- This does not currently exist in the makehuman model
 "_toe2-3_L", // 468 "toe2-3.L", <- This does not currently exist in the makehuman model
 "_toe3-1_L", // 471 "toe3-1.L", <- This does not currently exist in the makehuman model
 "_toe3-2_L", // 474 "toe3-2.L", <- This does not currently exist in the makehuman model
 "_toe3-3_L", // 477 "toe3-3.L", <- This does not currently exist in the makehuman model
 "_toe4-1_L", // 480 "toe4-1.L", <- This does not currently exist in the makehuman model
 "_toe4-2_L", // 483 "toe4-2.L", <- This does not currently exist in the makehuman model
 "_toe4-3_L", // 486 "toe4-3.L", <- This does not currently exist in the makehuman model
 "_toe5-1_L", // 489 "toe5-1.L", <- This does not currently exist in the makehuman model
 "_toe5-2_L", // 492 "toe5-2.L", <- This does not currently exist in the makehuman model
 "_toe5-3_L", // 495 "toe5-3.L" <- This does not currently exist in the makehuman model
 };*/


int drawObjectAT(
                 GLuint programID,
                 GLuint vao,
                 GLuint MatrixID,
                 unsigned int triangleCount,
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
                                     1.0,//scaleX,
                                     1.0,//scaleY,
                                     1.0//scaleZ
                                    );

       return drawVertexArrayWithMVPMatrices(
                                             programID,
                                             vao,
                                             MatrixID,
                                             triangleCount,
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



int doOGLDrawing(
                 int programID,
                 GLuint MVPMatrixID ,
                 struct pose6D * eyePose,
                 GLuint eyeVao,
                 unsigned int eyeTriangleCount,
                 struct pose6D * humanPose,
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
                              1000.0,      //Far
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
     if (eyePose->usePoseMatrixDirectly)
     {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     eyeVao,
                                     MVPMatrixID,
                                     eyeTriangleCount,
                                     //-------------
                                     &eyePose->m,
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
                  eyeVao,
                  MVPMatrixID,
                  eyeTriangleCount,
                  //-------------
                  eyePose->x,
                  eyePose->y, //+0.05,
                  eyePose->z, //+0.8,
                  eyePose->roll,
                  eyePose->pitch,
                  eyePose->yaw,
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
     if (humanPose->usePoseMatrixDirectly)
     {
      drawVertexArrayWithMVPMatrices(
                                     programID,
                                     humanVao,
                                     MVPMatrixID,
                                     humanTriangleCount,
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
                  humanVao,
                  MVPMatrixID,
                  humanTriangleCount,
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
                           GLuint * renderedDepth
                         )
{
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../../shaders/TransformVertexShader.vertexshader", "../../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../../shaders/simple.vert", "../../../shaders/simple.frag");
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
                int renderForever
             )
{
 	// Get a handle for our "MVP" uniform
	GLuint MVPMatrixID = glGetUniformLocation(programID, "MVP");
    //------------------------------------------------------------------------------------
    GLuint eyeVAO;
    GLuint eyeArrayBuffer;
    unsigned int eyeTriangleCount  =  (unsigned int)  eyeModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &eyeVAO,
                             &eyeArrayBuffer,
                             programID,
                             eyeModel->vertices       ,  eyeModel->header.numberOfVertices      * sizeof(float),
                             eyeModel->normal         ,  eyeModel->header.numberOfNormals       * sizeof(float),
                             eyeModel->textureCoords  ,  eyeModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             eyeModel->colors         ,  eyeModel->header.numberOfColors        * sizeof(float),
                             humanModel->indices      ,  humanModel->header.numberOfIndices     * sizeof(unsigned int)//0,0 //Not Indexed
                          );
    //------------------------------------------------------------------------------------
    GLuint humanVAO;
    GLuint humanArrayBuffer;
    unsigned int humanTriangleCount  =  (unsigned int)  humanModel->header.numberOfVertices/3;
    pushObjectToBufferData(
                             1,
                             &humanVAO,
                             &humanArrayBuffer,
                             programID,
                             humanModel->vertices       ,  humanModel->header.numberOfVertices      * sizeof(float),
                             humanModel->normal         ,  humanModel->header.numberOfNormals       * sizeof(float),
                             humanModel->textureCoords  ,  humanModel->header.numberOfTextureCoords * sizeof(float),      //0,0 //No Texture
                             humanModel->colors         ,  humanModel->header.numberOfColors        * sizeof(float),
                             humanModel->indices        ,  humanModel->header.numberOfIndices       * sizeof(unsigned int)//0,0 //Not Indexed
                          );
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

        doOGLDrawing(
                     programID,
                     MVPMatrixID,
                     humanPose,
                     eyeVAO,
                     eyeTriangleCount,
                     humanPose,
                     humanVAO,
                     humanTriangleCount,
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
	glDeleteBuffers(1, &humanArrayBuffer);
	glDeleteBuffers(1, &eyeArrayBuffer);
	glDeleteVertexArrays(1, &humanVAO);
	glDeleteVertexArrays(1, &eyeVAO);
	return 1;
}


void wipeTRITransform(struct TRI_Model * model,const char * boneName)
{
   TRIBoneID boneIDToWipe=0;
   if ( findTRIBoneWithName(model,boneName,&boneIDToWipe) )
       {
         create4x4FIdentityMatrixDirect(&model->bones[boneIDToWipe].info->matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose);
         create4x4FIdentityMatrixDirect(&model->bones[boneIDToWipe].info->localTransformation);
         create4x4FIdentityMatrixDirect(&model->bones[boneIDToWipe].info->finalVertexTransformation);
       }
       else { fprintf(stderr,RED "Unable to find bone `%s` in %s\n" NORMAL,boneName,model->name); }
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

  GLuint programID=0;
  GLuint programFrameBufferID=0;
  GLuint FramebufferName=0;
  GLuint renderedTexture=0;
  GLuint renderedDepth=0;

  if (!initializeOGLRenderer(&programID,&programFrameBufferID,&FramebufferName,&renderedTexture,&renderedDepth))
  {
		fprintf(stderr, "Failed to initialize Shaders\n");
	 	return 1;
  }

   #define defaultModelToLoad "makehuman.tri"
   const char * modelToLoad = defaultModelToLoad;

   //------------------------------------------------------
   struct BVH_MotionCapture mc = {0};
   //------------------------------------------------------
   struct TRI_Model eyeModel={0};
   struct TRI_Model indexedEyeModel={0};
   //------------------------------------------------------
   struct pose6D humanPose={0};
   struct TRI_Model humanModel={0};
   struct TRI_Model indexedHumanModel={0};
   //------------------------------------------------------


   //Set human pose to somewhere visible..
   //-------------------------------------------------------------------
   humanPose.roll=120.0;//(float)  (rand()%90);
   humanPose.pitch=20.0;//(float) (rand()%90);
   humanPose.yaw=0.0;//(float)   (rand()%90);
   //-------------------------------------------------------------------
   humanPose.x=0.0f;//(float)  (1000-rand()%2000);
   humanPose.y=-7.976f;//(float) (100-rand()%200);
   humanPose.z=27.99735f;//(float)  (700+rand()%1000);
   //-------------------------------------------------------------------

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
   //merged_neutral.bvh
   //char * defaultBVHFile = "merged_neutral.bvh";
   char * defaultBVHFile = "01_02.bvh";
   if (!bvh_loadBVH(defaultBVHFile,&mc,1.0) ) // This is the new armature that includes the head
        {
          fprintf(stderr,"Cannot find the merged_neutral.bvh file..\n");
          return 0;
        }
   //------------------------------------------------------
   if (!loadModelTri(modelToLoad, &indexedHumanModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/makehuman.tri\n");
     return 0;
   }
   //------------------------------------------------------
   if (!loadModelTri("eyes.tri", &indexedEyeModel ) )
   {
     fprintf(stderr,"Please : wget http://ammar.gr/mocapnet/eyes.tri\n");
     return 0;
   }
   //------------------------------------------------------


   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedEyeModel);
   removePrefixFromAllTRIBoneNames(&indexedEyeModel,"test_"); //Eyes have a test_ prefix on bone names..
   //------------------------------------------------------

   makeAllTRIBoneNamesLowerCaseWithoutUnderscore(&indexedHumanModel);
   //------------------------------------------------------


   //Wipe scene transformations
   //wipeTRITransform(&indexedHumanModel,"scene");
   //wipeTRITransform(&indexedEyeModel,"scene");

   //indexedHumanModel.header.rootBone=0;
   //indexedEyeModel.header.rootBone=0;

   printTRIBoneStructure(&indexedHumanModel,0 /*alsoPrintMatrices*/);
   bvh_printBVH(&mc);

   const int staticRendering = 0;

   while (1)
   {
    for (BVHFrameID fID=0; fID<mc.numberOfFrames; fID++)
    {
     fprintf(stderr,CYAN "\nBVH %s Frame %u/%u \n" NORMAL,mc.fileName,fID,mc.numberOfFrames);
     //-------------------------------------------

     if (!staticRendering)
     {
       //First animate the tri model using the bvh armature
       animateTRIModelUsingBVHArmature(&humanModel,&indexedHumanModel,&mc,fID,0);
       //triDeepCopyBoneValuesButNotStructure(&indexedEyeModel,&indexedHumanModel);

       //The eyes model should now have correct bone structure..
       animateTRIModelUsingBVHArmature(&eyeModel,&indexedEyeModel,&mc,fID,0);
     } else
     {
       fillFlatModelTriFromIndexedModelTri(&humanModel,&indexedHumanModel);
       fillFlatModelTriFromIndexedModelTri(&eyeModel,&indexedEyeModel);
     }


     doDrawing(
                programID,
                programFrameBufferID,
                FramebufferName,
                renderedTexture,
                renderedDepth,
                &humanPose,
                &humanModel,
                &eyeModel,
                0
              );

     deallocInternalsOfModelTri(&humanModel);
     deallocInternalsOfModelTri(&eyeModel);
     usleep(10);
    }
     fprintf(stderr,CYAN "\n\nLooping Dataset\n\n" NORMAL);
   }

   glDeleteProgram(programID);

   stop_glx3_stuff();
 return 0;
}

