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


#include "glx3.h"

#include "../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../Rendering/ShaderPipeline/shader_loader.h"

#define U 3.5
#define BUFFER_OFFSET( offset )   ((GLvoid*) (offset))


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


  int WIDTH=640;
  int HEIGHT=480;

unsigned int NumVertices=0;
	GLuint vao=0;

float cubeCoords[]=
{
-U,-U,-U,
-U,-U, U,
-U, U, U,
 U, U,-U,
-U,-U,-U,
-U, U,-U,
 U,-U, U,
-U,-U,-U,
 U,-U,-U,
 U, U,-U,
 U,-U,-U,
-U,-U,-U,
-U,-U,-U,
-U, U, U,
-U, U,-U,
 U,-U, U,
-U,-U, U,
-U,-U,-U,
-U, U, U,
-U,-U, U,
 U,-U, U,
 U, U, U,
 U,-U,-U,
 U, U,-U,
 U,-U,-U,
 U, U, U,
 U,-U, U,
 U, U, U,
 U, U,-U,
-U, U,-U,
 U, U, U,
-U, U,-U,
-U, U, U,
 U, U, U,
-U, U, U,
 U,-U, U
 };

float cubeNormals[]={ //X  Y  Z  W
                      -1.0f,-0.0f,-0.0f,
                      -1.0f,-0.0f,-0.0f,
                      -1.0f,-0.0f,-0.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                      -1.0f,-0.0f,0.0f,
                      -1.0f,-0.0f,0.0f,
                      -1.0f,-0.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,-0.0f,0.0f,
                       1.0f,-0.0f,0.0f,
                       1.0f,-0.0f,0.0f,
                       0.0f,1.0f,0.0f,
                       0.0f,1.0f,0.0f,
                       0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f
};


static const GLfloat g_vertex_buffer_data[] = {
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		-1.0f,-1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		-1.0f,-1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f,-1.0f,
		 1.0f,-1.0f,-1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f,-1.0f,
		 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f,-1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,
		 1.0f,-1.0f, 1.0f
	};

	// One color for each vertex. They were generated randomly.
	static const GLfloat g_color_buffer_data[] = {
		0.583f,  0.771f,  0.014f,
		0.609f,  0.115f,  0.436f,
		0.327f,  0.483f,  0.844f,
		0.822f,  0.569f,  0.201f,
		0.435f,  0.602f,  0.223f,
		0.310f,  0.747f,  0.185f,
		0.597f,  0.770f,  0.761f,
		0.559f,  0.436f,  0.730f,
		0.359f,  0.583f,  0.152f,
		0.483f,  0.596f,  0.789f,
		0.559f,  0.861f,  0.639f,
		0.195f,  0.548f,  0.859f,
		0.014f,  0.184f,  0.576f,
		0.771f,  0.328f,  0.970f,
		0.406f,  0.615f,  0.116f,
		0.676f,  0.977f,  0.133f,
		0.971f,  0.572f,  0.833f,
		0.140f,  0.616f,  0.489f,
		0.997f,  0.513f,  0.064f,
		0.945f,  0.719f,  0.592f,
		0.543f,  0.021f,  0.978f,
		0.279f,  0.317f,  0.505f,
		0.167f,  0.620f,  0.077f,
		0.347f,  0.857f,  0.137f,
		0.055f,  0.953f,  0.042f,
		0.714f,  0.505f,  0.345f,
		0.783f,  0.290f,  0.734f,
		0.722f,  0.645f,  0.174f,
		0.302f,  0.455f,  0.848f,
		0.225f,  0.587f,  0.040f,
		0.517f,  0.713f,  0.338f,
		0.053f,  0.959f,  0.120f,
		0.393f,  0.621f,  0.362f,
		0.673f,  0.211f,  0.457f,
		0.820f,  0.883f,  0.371f,
		0.982f,  0.099f,  0.879f
	};


void printOpenGLError(int errorCode)
{
  switch (errorCode)
  {
    case  GL_NO_ERROR       :
         fprintf(stderr,"No error has been recorded.");
        break;
    case  GL_INVALID_ENUM   :
         fprintf(stderr,"An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.\n");
        break;
    case  GL_INVALID_VALUE  :
         fprintf(stderr,"A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.");
        break;
    case  GL_INVALID_OPERATION :
         fprintf(stderr,"The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.");
        break;
    case  GL_INVALID_FRAMEBUFFER_OPERATION :
         fprintf(stderr,"The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag.");
        break;
    case  GL_OUT_OF_MEMORY :
         fprintf(stderr,"There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded.");
        break;
    case  GL_STACK_UNDERFLOW :
         fprintf(stderr,"An attempt has been made to perform an operation that would cause an internal stack to underflow.");
        break;
    case  GL_STACK_OVERFLOW :
         fprintf(stderr,"An attempt has been made to perform an operation that would cause an internal stack to overflow.");
     break;
  };
}


int checkOpenGLError(char * file , int  line)
{
  int err=glGetError();
  if (err !=  GL_NO_ERROR /*0*/ )
    {
      fprintf(stderr,RED "OpenGL Error (%u) : %s %u \n ", err , file ,line );
      printOpenGLError(err);
      fprintf(stderr,"\n" NORMAL);
      return 1;
    }
 return 0;
}

int drawGenericTriangleMesh(float * coords , float * normals, unsigned int coordLength)
{

    glBegin(GL_TRIANGLES);
      unsigned int i=0,z=0;
      for (i=0; i<coordLength/3; i++)
        {
                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z=(i*3)*3;  glVertex3f(coords[z+0],coords[z+1],coords[z+2]);

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(coords[z+0],coords[z+1],coords[z+2]);

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(coords[z+0],coords[z+1],coords[z+2]);
        }
    glEnd();
    return 1;
}

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
                             GLuint programID  ,
                             const float * vertices , unsigned int verticesLength ,
                             const float * normals , unsigned int normalsLength ,
                             const float * colors , unsigned int colorsLength
                           )
{
    glGenVertexArrays(1, &vao);
    checkOpenGLError(__FILE__, __LINE__);
    glBindVertexArray(vao);
    checkOpenGLError(__FILE__, __LINE__);


    // Create and initialize a buffer object on the server side (GPU)
    GLuint      buffer;
    glGenBuffers( 1, &buffer );
    checkOpenGLError(__FILE__, __LINE__);
    glBindBuffer( GL_ARRAY_BUFFER, buffer );

    NumVertices+=(unsigned int ) verticesLength/(3*sizeof(float));
    fprintf(stderr,"Will DrawArray(GL_TRIANGLES,0,%u) - %u \n"  ,NumVertices,verticesLength);
    fprintf(stderr,"Pushing %u vertices (%u bytes) and %u normals (%u bytes) as our object \n"  ,verticesLength/sizeof(float),verticesLength,normalsLength/sizeof(float),normalsLength);
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

    GLuint vPosition = glGetAttribLocation( programID, "vPosition" );
     checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vPosition );
     checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(0) );
    checkOpenGLError(__FILE__, __LINE__);

    /*
    GLuint vNormal = glGetAttribLocation( programID, "vNormal" );
    checkOpenGLError(__FILE__, __LINE__);
     glEnableVertexAttribArray( vNormal );
    checkOpenGLError(__FILE__, __LINE__);
     glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(verticesLength) );
     checkOpenGLError(__FILE__, __LINE__);*/


    if ( (colors!=0) && (colorsLength!=0) )
    {
     GLuint vColor = glGetAttribLocation( programID, "vColor" );
     glEnableVertexAttribArray( vColor );
     glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( verticesLength + normalsLength ) );
     checkOpenGLError(__FILE__, __LINE__);
    }

  return buffer;
}





int doDrawing()
{
   fprintf(stderr," doDrawing \n");


	// Create and compile our GLSL program from the shaders
    fprintf(stderr," loadShader \n");
	//struct shaderObject * sho = loadShader("../../shaders/TransformVertexShader.vertexshader", "../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../shaders/simple.vert", "../../shaders/simple.frag");
    if (sho==0) { fprintf(stderr,"Could not load..\n"); exit(1); }
    GLuint programID = sho->ProgramObject;


     double projectionMatrixD[16];
     int viewport[4]={0};
     double fx = 535.423889;
     double fy = 533.48468;
     double skew = 0.0;
     double cx = (double) WIDTH/2;
     double cy = (double) HEIGHT/2;
     buildOpenGLProjectionForIntrinsics(
                                        projectionMatrixD ,
                                         viewport ,
                                         fx, fy,
                                         skew,
                                         cx,  cy,
                                         WIDTH, HEIGHT,
                                         0.1,
                                         1000
                                         );

     //glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);



     double modelViewMatrixD[16];
     create4x4ModelTransformation(
                                  modelViewMatrixD,
                                  //Rotation Component
                                  0.5 ,//roll
                                  0.0 ,//pitch
                                  0.0 ,//yaw

                                  //Translation Component
                                  0.0  ,//X
                                  0.0  ,//Y
                                  100 ,//Z

                                  1.0,//scaleX,
                                  1.0,//scaleY,
                                  1.0//scaleZ
                                 );


    double MVPD[16];
    multiplyTwo4x4Matrices(MVPD,projectionMatrixD,modelViewMatrixD);

    float MVP[16]={
                   -1.09,0.00,-1.45,0.00,
                   -0.99,2.07,0.75,0.00,
                   -0.69,-0.52,0.52,5.64,
                   -0.69,-0.51,0.51,5.83
                  };

   copy4x4DMatrixToF(MVP , MVPD );
   transpose4x4Matrix(MVP);

 	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");


    float * vertices = cubeCoords;
    float * normals = cubeNormals;
    unsigned int numberOfVertices = sizeof(cubeCoords);
    unsigned int numberOfNormals = sizeof(cubeNormals);
    unsigned int wireFrame=0;




		// Use our shader
		glUseProgram(programID);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	GLuint VertexArrayID=0;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

   fprintf(stderr,"Ready to start rendering : ");


		pushObjectToBufferData(
                                 programID  ,
                                 vertices ,  numberOfVertices ,
                                 normals ,  numberOfNormals ,
                                 0 , 0
                              );



	do{

       glUseProgram(programID);
       checkOpenGLError(__FILE__, __LINE__);

       glBindVertexArray(vao);
        checkOpenGLError(__FILE__, __LINE__);

        fprintf(stderr,".");
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, MVP);



        glPushAttrib(GL_ALL_ATTRIB_BITS);
          glDisable(GL_CULL_FACE);




         if (wireFrame)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        checkOpenGLError(__FILE__, __LINE__);
        glDrawArrays( GL_TRIANGLES, 0, NumVertices );
        checkOpenGLError(__FILE__, __LINE__);

       glPopAttrib();
       glBindVertexArray(0);
       checkOpenGLError(__FILE__, __LINE__);







		// Swap buffers
        glx3_endRedraw();

	} // Check if the ESC key was pressed or the window was closed
	while( 1 );

	// Cleanup VBO and shader
	//glDeleteBuffers(1, &vertexbuffer);
	//glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);

}














int main(int argc, char **argv)
{
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);


  glClearColor ( 1, 0.5, 0, 1 );
  glClear ( GL_COLOR_BUFFER_BIT );
  glMatrixMode(GL_MODELVIEW );


  //glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
  //glGetFloatv( GL_PROJECTION_MATRIX, projection );
  //glGetIntegerv( GL_VIEWPORT, viewport );
  glViewport(0,0,WIDTH,HEIGHT);

  glx3_endRedraw();

  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 1;
   }



  doDrawing();
/*
  while (1)
   {
     glx3_checkEvents();
     fprintf(stderr,".");
     drawGenericTriangleMesh(cubeCoords,cubeNormals,36*3);
     usleep(100);
     glx3_endRedraw();
   }*/


  stop_glx3_stuff();
 return 0;
}
