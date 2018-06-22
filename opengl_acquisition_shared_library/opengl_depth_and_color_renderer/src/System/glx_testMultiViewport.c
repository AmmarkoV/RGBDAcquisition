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

#include "glx3.h"

#include "../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../tools/AmMatrix/matrixOpenGL.h"

#include "../Rendering/ShaderPipeline/shader_loader.h"
#include "../Rendering/ShaderPipeline/render_buffer.h"

#define BUFFER_OFFSET( offset )   ((GLvoid*) (offset))


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


int WIDTH=640;
int HEIGHT=480;
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



GLuint
pushObjectToBufferData(
                             GLuint *vao ,
                             GLuint programID  ,
                             const float * vertices , unsigned int verticesLength ,
                             const float * normals , unsigned int normalsLength ,
                             const float * colors , unsigned int colorsLength
                           )
{
    glGenVertexArrays(1, vao);              checkOpenGLError(__FILE__, __LINE__);
    glBindVertexArray(*vao);                checkOpenGLError(__FILE__, __LINE__);


    // Create and initialize a buffer object on the server side (GPU)
    GLuint      buffer;
    glGenBuffers( 1, &buffer );                 checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, buffer );    checkOpenGLError(__FILE__, __LINE__);

    unsigned int NumVertices=(unsigned int ) verticesLength/(3*sizeof(float));
    fprintf(stderr,"Will DrawArray(GL_TRIANGLES,0,%u) - %u \n"  ,NumVertices,verticesLength);
    fprintf(stderr,"Pushing %lu vertices (%u bytes) and %u normals (%u bytes) as our object \n"  ,verticesLength/sizeof(float),verticesLength,normalsLength/sizeof(float),normalsLength);
    glBufferData( GL_ARRAY_BUFFER, verticesLength + normalsLength  + colorsLength  ,NULL, GL_STATIC_DRAW );   checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, 0                                      , verticesLength , vertices );   checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, verticesLength                         , normalsLength  , normals );    checkOpenGLError(__FILE__, __LINE__);

    if ( (colors!=0) && (colorsLength!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, verticesLength + normalsLength , colorsLength , colors );               checkOpenGLError(__FILE__, __LINE__);
    }

    GLuint vPosition = glGetAttribLocation( programID, "vPosition" );               checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vPosition );                                         checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(0) );  checkOpenGLError(__FILE__, __LINE__);


    GLuint vNormal = glGetAttribLocation( programID, "vNormal" );                             checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vNormal );                                                     checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(verticesLength) ); checkOpenGLError(__FILE__, __LINE__);


    if ( (colors!=0) && (colorsLength!=0) )
    {
     GLuint vColor = glGetAttribLocation( programID, "vColor" );
     glEnableVertexAttribArray( vColor );
     glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( verticesLength + normalsLength ) );
     checkOpenGLError(__FILE__, __LINE__);
    }

  return buffer;
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








int doDrawing()
{
   fprintf(stderr," doDrawing \n");
	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../shaders/TransformVertexShader.vertexshader", "../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../shaders/simple.vert", "../../shaders/simple.frag");
	if (sho==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

	struct shaderObject * textureFramebuffer = loadShader("../../shaders/virtualFramebuffer.vert", "../../shaders/virtualFramebufferSea.frag");
    if (textureFramebuffer==0) {  checkOpenGLError(__FILE__, __LINE__); exit(1); }

    GLuint programID = sho->ProgramObject;
    GLuint programFrameBufferID = textureFramebuffer->ProgramObject;





     double projectionMatrixD[16];
     double viewportMatrixD[16];
     double viewMatrixD[16];

     prepareRenderingMatrices(
                     535.423889, //fx
                     533.48468,  //fy
                     0.0,        //skew
                     WIDTH/2,    //cx
                     HEIGHT/2,   //cy
                     WIDTH,      //Window Width
                     HEIGHT,     //Window Height
                     1.0,        //Near
                     255.0,      //Far
                     projectionMatrixD,
                     viewMatrixD,
                     viewportMatrixD
                    );



 	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");


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
    unsigned int cubeTriangleCount  =  (unsigned int )  sizeof(cubeCoords)/(3*sizeof(float));
    pushObjectToBufferData(
                                 &cubeVAO,
                                 programID   ,
                                 cubeCoords  ,  sizeof(cubeCoords) ,
                                 cubeNormals ,  sizeof(cubeNormals) ,
                                 cubeColors  , sizeof(cubeColors)
                              );



    GLuint pyramidVAO;
    unsigned int pyramidTriangleCount  =  (unsigned int )  sizeof(pyramidCoords)/(3*sizeof(float));
    pushObjectToBufferData(
                                 &pyramidVAO,
                                 programID  ,
                                 pyramidCoords ,  sizeof(pyramidCoords) ,
                                 pyramidNormals ,  sizeof(pyramidNormals) ,
                                 cubeColors , sizeof(pyramidCoords)
                              );



    fprintf(stderr,"Ready to render: ");

     //-------------------------------------------------------------------
        double roll=0.0;//(double)  (rand()%90);
        double pitch=0.0;//(double) (rand()%90);
        double yaw=0.0;//(double)   (rand()%90);

        double x=-259.231f;//(double)  (1000-rand()%2000);
        double y=-54.976f;//(double) (100-rand()%200);
        double z=2699.735f;//(double)  (700+rand()%1000);
     //-------------------------------------------------------------------


     GLuint FramebufferName = 0;
     GLuint renderedTexture;
     GLuint renderedDepth;
     initializeFramebuffer(&FramebufferName,&renderedTexture,&renderedDepth);

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

       roll+=1.0;
       pitch+=1.5;

     drawObjectAT(
                  programID,
                  cubeVAO,
                  MatrixID,
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
                  pyramidVAO,
                  MatrixID,
                  pyramidTriangleCount,
                  x+1100,
                  y,
                  z,
                  roll,
                  pitch,
                  yaw,

                  projectionMatrixD,
                  viewportMatrixD,
                  viewMatrixD
                 );


        //We have accumulated all data on the framebuffer and will now draw it back..
        drawFramebuffer(
                        programFrameBufferID,
                        quad_vertexbuffer,
                        //renderedDepth,
                        renderedTexture,
                        texID,
                        timeID,
                        resolutionID
                       );


		// Swap buffers
        glx3_endRedraw();
        usleep(5);

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
	glDeleteVertexArrays(1, &pyramidVAO);
	glDeleteVertexArrays(1, &cubeVAO);
}





int main(int argc, char **argv)
{
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }



   doDrawing();

  stop_glx3_stuff();
 return 0;
}

