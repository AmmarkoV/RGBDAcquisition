//
//  main.cpp
//  basicCube
//
//  Created by George Papagiannakis on 23/10/12.
//  Copyright (c) 2012 University Of Crete & FORTH. All rights reserved.
//


// basic STL streams
#include <iostream>

// GLEW lib
// http://glew.sourceforge.net/basic.html
#include <GL/glew.h>

// Here we decide which of the two versions we want to use
// If your systems supports both, choose to uncomment USE_OPENGL32
// otherwise choose to uncomment USE_OPENGL21
// GLView cna also help you decide before running this program:
//
// FOR MACOSX only, please use OPENGL32 for AntTweakBar to work properly
//
#define USE_OPENGL32
//#define USE_OPENGL21


#define USE_SIMPLE_SHADERS 0

#ifdef USE_OPENGL21
#ifdef __APPLE__
#define glGenVertexArrays glGenVertexArraysAPPLE
#define glBindVertexArray glBindVertexArrayAPPLE
#define glDeleteVertexArray glDeleteVertexArrayAPPLE
#endif
#endif //USE_OPENGL21

// GLFW lib
// http://www.glfw.org/documentation.html
#ifdef USE_OPENGL32
    #ifndef _WIN32
        #define GLFW_INCLUDE_GL3
        #define USE_GL3
        #define GLFW_NO_GLU
        #define GL3_PROTOTYPES 1
    #endif
#endif
#define GLFW_DLL //use GLFW as a dynamically linked library
#include <GL/glfw.h>

// GLM lib
// http://glm.g-truc.net/api/modules.html
#define GLM_SWIZZLE
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

//local
#include "glGA/glGAHelper.h"

//GUI AntTweakBar
#include <AntTweakBar/AntTweakBar.h>



#include "scene.h"
#include "controls.h"
#include "matrices.h"
#include "tools.h"
#include "quaternions.h"


#include "hardcoded.h"

// global variables


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


int autoRotate=1;
int autoDirection=1;

float fx=535.423889;
float fy=533.484680;
float fov=45;
float near=1.00;
float far=1000.00;

int UseGLMMatrices=1;

int         windowWidth=640, windowHeight=480;
GLuint      program;
GLuint      vao=0;
GLuint      buffer;
int        wireFrame=0;
// GUI TweakBar variables
TwBar*      myBar    = NULL;
glm::vec4           bgColor(0.0f,0.0f,0.0f,1.0f);
// model related variables
int                     Index = 0;
unsigned int         NumVertices = 0; //(6 faces)(2 triangles/face)(3 vertices/triangle)

 GLuint vPosition , vNormal , vColor  , lightPositionLocation  , fogLocation  , modelViewMatrixLocation  , modelViewProjectionMatrixLocation  , normalTransformationLocation;
 GLuint lightColorLocation , lightMaterialsLocation    ;



// API function prototypes
void GLFWCALL   WindowSizeCB(int, int);

// Callback function called by GLFW when window size changes
void GLFWCALL WindowSizeCB(int width, int height)
{
    // Set OpenGL viewport and default camera
    glViewport(0, 0, width, height);

    // Send the new window size to AntTweakBar
    TwWindowSize(width, height);
}



void glmMatToFloat(float * out , glm::mat4 in)
{
  unsigned int i=0;
  const float *pSource = (const float*)glm::value_ptr(in);

  for (i=0; i<16; i++)
  {
      out[i]=pSource[i];
  }
}


void updateViewMy(int uploadIt)
{

    int viewport[4]={0};
    glm::mat4 Projection = glm::perspective(fov, 1.0f, near , far);

    buildOpenGLProjectionForIntrinsics   (
                                             projectionMatrix ,
                                             viewport ,
                                             fx,
                                             fy,
                                             0.0 , // SKEW
                                             windowWidth/2.0, windowHeight/2.0,
                                             windowWidth, windowHeight,
                                             near,
                                             far
                                           );

    glmMatToFloat(projectionMatrix, Projection);

    float rotation[16];
    float translation[16];
    MatrixF4x42Quaternion(camera.angle,qXqYqZqW,rotation);
    create4x4IdentityFMatrix(rotation);

    create4x4TranslationMatrix(translation , camera.pos[0] , camera.pos[1] , camera.pos[2]);

    multiplyTwo4x4FMatrices(modelViewMatrix, translation , rotation );

    multiplyTwo4x4FMatrices(modelViewProjectionMatrix, modelViewMatrix , projectionMatrix  );
    transpose4x4MatrixF(modelViewMatrix);
    transpose4x4MatrixF(modelViewProjectionMatrix);

    if (uploadIt)
    {
	 glUniformMatrix4fv(modelViewProjectionMatrixLocation  , 1 /*Only setting one matrix*/ , GL_FALSE /* dont transpose */, (const float * ) modelViewProjectionMatrix);
     checkOpenGLError(__FILE__, __LINE__);
    }
}




void updateViewGLM(int uploadIt)
{
    glm::mat4  GLMprojectionMatrix = glm::perspective(fov, (float)windowWidth/windowHeight, near, far);

    glm::mat4  GLMModelViewMatrix       = glm::lookAt( glm::vec3(camera.pos[0],camera.pos[1],camera.pos[2]), // Camera is at (4,3,3), in World Space
                                                            glm::vec3(cameraLookAt.pos[0],cameraLookAt.pos[1],cameraLookAt.pos[2]), // and looks at the origin
                                                            glm::vec3(0,1,0));  // Head is up (set to 0,-1,0 to look upside-do
    glm::mat4 MVP        =  GLMprojectionMatrix * GLMModelViewMatrix ; //*GLMmodelMatrix

    glm::mat4  GLMnormalMatrix = glm::inverse(GLMModelViewMatrix);
    glm::mat4  GLMnormalTMatrix = glm::transpose(GLMnormalMatrix);


    glm::vec4   lightInitial( lightPosition[0], lightPosition[1] , lightPosition[2] , 1.0 );
    glm::vec4   GLMlightPos=GLMModelViewMatrix * lightInitial ;
    float       lightPosFinal[3];
    lightPosFinal[0]=GLMlightPos[0]; lightPosFinal[1]=GLMlightPos[1]; lightPosFinal[2]=GLMlightPos[2];

    if (uploadIt)
    {
	 glUniform3fv(lightPositionLocation, 1, &lightPosFinal[0]);
     checkOpenGLError(__FILE__, __LINE__);
     glUniformMatrix4fv(modelViewProjectionMatrixLocation , 1 /*Only setting one matrix*/ , GL_FALSE /* dont transpose */, &MVP[0][0]);
     checkOpenGLError(__FILE__, __LINE__);
	 glUniformMatrix4fv(modelViewMatrixLocation , 1 /*Only setting one matrix*/ , GL_FALSE /* dont transpose */, &GLMModelViewMatrix[0][0]);
     checkOpenGLError(__FILE__, __LINE__);
	 glUniformMatrix4fv(normalTransformationLocation , 1 /*Only setting one matrix*/ , GL_FALSE /* dont transpose */, &GLMnormalTMatrix[0][0]);
     checkOpenGLError(__FILE__, __LINE__);
    }
}

void updateView(int uploadIt)
{
  checkOpenGLError(__FILE__, __LINE__);
  if (UseGLMMatrices)
  {
      updateViewGLM(uploadIt);
  } else
  {
      updateViewMy(uploadIt);
  }

  if (uploadIt)
  {
   glUniform4fv(lightColorLocation   , 1, &lightColor[0]);
   checkOpenGLError(__FILE__, __LINE__);
   glUniform4fv(fogLocation , 1, &fogColorAndScale[0]);
   checkOpenGLError(__FILE__, __LINE__);
   glUniform4fv(lightMaterialsLocation  , 1, &lightMaterials[0]);
   checkOpenGLError(__FILE__, __LINE__);
  }
}


void pushObjectToBufferData(
                             const float * vertices , unsigned int verticesLength ,
                             const float * normals , unsigned int normalsLength ,
                             const float * colors , unsigned int colorsLength
                           )
{
    NumVertices+=(unsigned int ) verticesLength/(3*sizeof(float));
    fprintf(stderr,GREEN "Will DrawArray(GL_TRIANGLES,0,%u)\n" NORMAL ,NumVertices);
    fprintf(stderr,GREEN "Pushing %u vertices (%u bytes) and %u normals (%u bytes) as our object \n" NORMAL ,verticesLength/sizeof(float),verticesLength,normalsLength/sizeof(float),normalsLength);
    glBufferData( GL_ARRAY_BUFFER, verticesLength + normalsLength  + colorsLength  ,NULL, GL_STATIC_DRAW );
     checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, 0                                      , verticesLength , vertices );
     checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, verticesLength                         , normalsLength  , normals );
     checkOpenGLError(__FILE__, __LINE__);

    if ( (colors!=0) && (colorsLength!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, verticesLength + normalsLength , colorsLength , colors );
     checkOpenGLError(__FILE__, __LINE__);
    }

    vPosition = glGetAttribLocation( program, "vPosition" );
     checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vPosition );
     checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(0) );
    checkOpenGLError(__FILE__, __LINE__);

     vNormal = glGetAttribLocation( program, "vNormal" );
    checkOpenGLError(__FILE__, __LINE__);
     glEnableVertexAttribArray( vNormal );
    checkOpenGLError(__FILE__, __LINE__);
     glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(verticesLength) );
     checkOpenGLError(__FILE__, __LINE__);


    if ( (colors!=0) && (colorsLength!=0) )
    {
     vColor = glGetAttribLocation( program, "vColor" );
     glEnableVertexAttribArray( vColor );
     glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( verticesLength + normalsLength ) );
     checkOpenGLError(__FILE__, __LINE__);
    }

}


void initScene()
{
    print4x4FMatrix("modelViewMatrix",modelViewMatrix);
    int viewport[4]={0};

    buildOpenGLProjectionForIntrinsics   (
                                             projectionMatrix ,
                                             viewport ,
                                             fx,
                                             fy,
                                             0.0 , // SKEW
                                             windowWidth/2.0, windowHeight/2.0,
                                             windowWidth, windowHeight,
                                             near,
                                             far
                                           );

    print4x4FMatrix("projectionMatrix",projectionMatrix);

    lightPosition[0]=2;
    lightPosition[1]=2;
    lightPosition[2]=2;

    //create4x4RotationMatrix(modelViewProjection , 30 , 1.0, 1.0 , 1.0 );
    camera.pos[0]=0;
    camera.pos[1]=0;
    camera.pos[2]=-3;
    camera.angle[0]=0; camera.angle[1]=0; camera.angle[2]=0; camera.angle[3]=0;
    updateView(0);

    //generate and bind a VAO for the 3D axes
    glGenVertexArrays(1, &vao);
    checkOpenGLError(__FILE__, __LINE__);
    glBindVertexArray(vao);
    checkOpenGLError(__FILE__, __LINE__);

    //colorcube();

    // Load shaders and use the resulting shader program
    #if USE_SIMPLE_SHADERS
      fprintf(stderr,GREEN "Loading simple shader set \n" NORMAL);
      program = LoadShaders( "simple.vert", "simple.frag" );
      checkOpenGLError(__FILE__, __LINE__);
      #else
      fprintf(stderr,GREEN "Loading complex shader set \n"  NORMAL);
       program = LoadShaders( "test.vert", "test.frag" );
       checkOpenGLError(__FILE__, __LINE__);
    #endif // USE_SIMPLE_SHADERS

    glUseProgram( program );
    checkOpenGLError(__FILE__, __LINE__);

    // Create and initialize a buffer object on the server side (GPU)
    glGenBuffers( 1, &buffer );
    checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, buffer );
    checkOpenGLError(__FILE__, __LINE__);
    NumVertices=0;
    //pushObjectToBufferData( heartVertices , sizeof(heartVertices) , heartNormals , sizeof(heartNormals) , 0 , 0 );
    //pushObjectToBufferData( cubeCoords , sizeof(cubeCoords) , cubeNormals , sizeof(cubeNormals) , 0 , 0 );
    //pushObjectToBufferData( planeCoords , sizeof(planeCoords) , planeNormals , sizeof(planeNormals) , 0 , 0 );
    pushObjectToBufferData( pyramidCoords , sizeof(pyramidCoords) , pyramidNormals , sizeof(pyramidNormals) , 0 , 0 );

    lightPositionLocation = glGetUniformLocation( program, "lightPosition" );                   checkOpenGLError(__FILE__, __LINE__);
    fogLocation = glGetUniformLocation( program, "fogColorAndScale" );                          checkOpenGLError(__FILE__, __LINE__);
    lightColorLocation = glGetUniformLocation( program, "lightColor" );                         checkOpenGLError(__FILE__, __LINE__);
    lightMaterialsLocation   = glGetUniformLocation( program, "lightMaterials" );               checkOpenGLError(__FILE__, __LINE__);
	normalTransformationLocation = glGetUniformLocation(program, "normalTransformation");       checkOpenGLError(__FILE__, __LINE__);
	modelViewProjectionMatrixLocation  = glGetUniformLocation(program, "modelViewProjection");  checkOpenGLError(__FILE__, __LINE__);
	modelViewMatrixLocation  = glGetUniformLocation(program, "modelViewMatrix");                checkOpenGLError(__FILE__, __LINE__);

    updateView(1);


    glEnable( GL_DEPTH_TEST );
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    checkOpenGLError(__FILE__, __LINE__);


    // only one VAO can be bound at a time, so disable it to avoid altering it accidentally
    glBindVertexArray(0);
    checkOpenGLError(__FILE__, __LINE__);

}





void drawScene()
{
    glUseProgram(program);
     checkOpenGLError(__FILE__, __LINE__);

    glBindVertexArray(vao);
    checkOpenGLError(__FILE__, __LINE__);

   if (autoRotate)
   {
     if (autoDirection)
     {
       camera.pos[0]-=0.05;
       camera.pos[1]-=0.05;
       camera.pos[2]-=0.05;
       if (camera.pos[0]<-6.0) {autoDirection=0;}
     } else
     {
       camera.pos[0]+=0.05;
       camera.pos[1]+=0.05;
       camera.pos[2]+=0.05;
       if (camera.pos[0]>6.0) {autoDirection=1;}
     }
   }

  checkOpenGLError(__FILE__, __LINE__);
  updateView(1);


    glPushAttrib(GL_ALL_ATTRIB_BITS);
       supressOpenGLError(); // checkOpenGLError(__FILE__, __LINE__);
    glDisable(GL_CULL_FACE);
        checkOpenGLError(__FILE__, __LINE__);




    if (wireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        checkOpenGLError(__FILE__, __LINE__);

    glDrawArrays( GL_TRIANGLES, 0, NumVertices );

        checkOpenGLError(__FILE__, __LINE__);

    glPopAttrib();
       supressOpenGLError(); // checkOpenGLError(__FILE__, __LINE__);
    glBindVertexArray(0);
        checkOpenGLError(__FILE__, __LINE__);

}

int main (int argc, const char * argv[])
{
    // initialise GLFW
    int running = GL_TRUE;

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwOpenWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);


    GLFWvidmode mode;
    glfwGetDesktopMode(&mode);
    if( !glfwOpenWindow(windowWidth, windowHeight, mode.RedBits, mode.GreenBits, mode.BlueBits, 0, 32, 0, GLFW_WINDOW /* or GLFW_FULLSCREEN */) )
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwEnable(GLFW_MOUSE_CURSOR);
    glfwEnable(GLFW_KEY_REPEAT);
    // Ensure we can capture the escape key being pressed below
	glfwEnable( GLFW_STICKY_KEYS );
	glfwSetMousePos(windowWidth/2, windowHeight/2);
    glfwSetWindowTitle("basic Cube with GUI");

    //init GLEW and basic OpenGL information
    // VERY IMPORTANT OTHERWISE GLEW CANNOT HANDLE GL3
#ifdef USE_OPENGL32
    glewExperimental = true;
#endif
    glewInit();
    std::cout<<"\nUsing GLEW "<<glewGetString(GLEW_VERSION)<<std::endl;
    if (GLEW_VERSION_3_2)
    {
        std::cout<<"Yay! OpenGL 3.2 is supported and GLSL 1.5!\n"<<std::endl;
    }

	if (glewIsSupported("GL_ARB_vertex_buffer_object"))
		std::cout<<"ARB VBO's are supported"<<std::endl;
    else if (glewIsSupported("GL_APPLE_vertex_buffer_object"))
		std::cout<<"APPLE VBO's are supported"<<std::endl;
	else
		std::cout<<"VBO's are not supported,program will not run!!!"<<std::endl;


	if (glewIsSupported("GL_ARB_vertex_array_object"))
        std::cout<<"ARB VAO's are supported\n"<<std::endl;
    else if (glewIsSupported("GL_APPLE_vertex_array_object"))//this is the name of the extension for GL2.1 in MacOSX
		std::cout<<"APPLE VAO's are supported\n"<<std::endl;
	else
		std::cout<<"VAO's are not supported, program will not run!!!\n"<<std::endl;


    std::cout<<"Vendor: "<<glGetString (GL_VENDOR)<<std::endl;
    std::cout<<"Renderer: "<<glGetString (GL_RENDERER)<<std::endl;
    std::cout<<"Version: "<<glGetString (GL_VERSION)<<std::endl;

    //init AntTweakBar
    TwInit(TW_OPENGL_CORE, NULL);   //OpenGL 3.2


    // Set GLFW event callbacks
    // - Redirect window size changes to the callback function WindowSizeCB
    glfwSetWindowSizeCallback(WindowSizeCB);

    // - Directly redirect GLFW mouse button events to AntTweakBar
    //glfwSetMouseButtonCallback((GLFWmousebuttonfun)TwEventMouseButtonGLFW);
    glfwSetMouseButtonCallback(OnMouseButton);

    // - Directly redirect GLFW mouse position events to AntTweakBar
    //glfwSetMousePosCallback((GLFWmouseposfun)TwEventMousePosGLFW);
    glfwSetMousePosCallback(OnMousePos);

    // - Directly redirect GLFW mouse wheel events to AntTweakBar
    //glfwSetMouseWheelCallback((GLFWmousewheelfun)TwEventMouseWheelGLFW);
    glfwSetMouseWheelCallback(OnMouseWheel);

    // - Directly redirect GLFW key events to AntTweakBar
    //glfwSetKeyCallback((GLFWkeyfun)TwEventKeyGLFW);
    glfwSetKeyCallback(OnKey);

    // - Directly redirect GLFW char events to AntTweakBar
    //glfwSetCharCallback((GLFWcharfun)TwEventCharGLFW);
    glfwSetCharCallback(OnChar);

    // Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);


    checkOpenGLError(__FILE__, __LINE__);
    // init Scene
    initScene();

    //Create tweak bars GUI
    TwWindowSize(windowWidth, windowHeight);
    myBar = TwNewBar("myBar");
        // Add 'wire' to 'bar': it is a modifable variable of type TW_TYPE_BOOL32 (32 bits boolean). Its key shortcut is [w].
    TwAddVarRW(myBar, "wire", TW_TYPE_BOOL32, &wireFrame, " label='Wireframe mode' key=w help='Toggle wireframe display mode.' ");
    TwAddVarRW(myBar, "useglm", TW_TYPE_BOOL32, &UseGLMMatrices               , " label='Use GLM' key=g help='Toggle using GLM for matrices.' ");
    TwAddVarRW(myBar, "autorotate", TW_TYPE_BOOL32, &autoRotate               , " label='AutoRotate' key=r help='Toggle autorotation.' ");




    // Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
    //TwAddVarRW(myBar, "bgColor", TW_TYPE_COLOR4F, &bgColor, " label='Background color' ");
    TwAddVarRW(myBar, "bgColor", TW_TYPE_COLOR4F, glm::value_ptr(bgColor), " label='Background color' ");


    TwAddVarRW(myBar, "fx", TW_TYPE_FLOAT, &fx , " min=0 max=1000 step=0.5 ");
    TwAddVarRW(myBar, "fy", TW_TYPE_FLOAT,   &fy , " min=0 max=1000 step=0.5 ");


    TwAddVarRW(myBar, "near", TW_TYPE_FLOAT,   &near , " min=0 max=1000 step=0.5 ");
    TwAddVarRW(myBar, "far", TW_TYPE_FLOAT ,   &far , " min=0 max=1000 step=0.5 ");

    TwAddVarRW(myBar, "CameraX", TW_TYPE_FLOAT, &camera.pos[0] , " min=-60 max=60 step=0.1 ");
    TwAddVarRW(myBar, "CameraY", TW_TYPE_FLOAT, &camera.pos[1], " min=-60 max=60 step=0.1 ");
    TwAddVarRW(myBar, "CameraZ", TW_TYPE_FLOAT, &camera.pos[2], " min=-60 max=60 step=0.1 ");

    TwAddVarRW(myBar, "LookX", TW_TYPE_FLOAT, &cameraLookAt.pos[0] , " min=-60 max=60 step=0.1 ");
    TwAddVarRW(myBar, "LookY", TW_TYPE_FLOAT, &cameraLookAt.pos[1], " min=-60 max=60 step=0.1 ");
    TwAddVarRW(myBar, "LookZ", TW_TYPE_FLOAT, &cameraLookAt.pos[2], " min=-60 max=60 step=0.1 ");



    TwAddVarRW(myBar, "Camera Rotation", TW_TYPE_QUAT4F, &camera.angle, "");



    TwAddVarRW(myBar, "LightX", TW_TYPE_FLOAT,   &lightPosition[0] , " min=-20 max=30 step=0.5 ");
    TwAddVarRW(myBar, "LightY", TW_TYPE_FLOAT,   &lightPosition[1] , " min=-20 max=30 step=0.5 ");
    TwAddVarRW(myBar, "LightZ", TW_TYPE_FLOAT,   &lightPosition[2] , " min=-20 max=30 step=0.5 ");



    TwAddVarRW(myBar, "lightColor", TW_TYPE_COLOR4F, lightColor, " label='Light color ' ");
    TwAddVarRW(myBar, "MaterialAmbient", TW_TYPE_FLOAT,   &lightMaterials[0] , " min=0 max=1.0 step=0.1 ");
    TwAddVarRW(myBar, "MaterialDiffuse", TW_TYPE_FLOAT,   &lightMaterials[1] , " min=0 max=1.0 step=0.1 ");
    TwAddVarRW(myBar, "MaterialSpecular", TW_TYPE_FLOAT,  &lightMaterials[2] , " min=0 max=1.0 step=0.1 ");




    TwAddVarRW(myBar, "fogColor", TW_TYPE_COLOR4F, fogColorAndScale, " label='Fog color and intensity' ");

    //TwDefine("myBar alwaystop=true "); // mybar is always on top


    //GLFW main loop
    while (running)
        {
         glClear( GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT );
         glClearColor( bgColor.r,bgColor.g,bgColor.b,bgColor.a);
         checkOpenGLError(__FILE__, __LINE__);


        // call function to render our scene
        drawScene();

        //restore FILL mode (we don't want the GUI to be drawn in 'wireframe')
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        checkOpenGLError(__FILE__, __LINE__);
        TwDraw();
         checkOpenGLError(__FILE__, __LINE__);

        glfwSwapBuffers();
        //check if ESC was pressed
        running=!glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED);
       }

    //terminate AntTweakBar
    TwTerminate();

    //cleanup VAO, VBO and shaders
    glDeleteBuffers(1,&buffer);
    glDeleteProgram(program);
    glDeleteVertexArrays(1,&vao);


    //close OpenGL window and terminate GLFW
    glfwTerminate();
    exit(EXIT_SUCCESS);

}



