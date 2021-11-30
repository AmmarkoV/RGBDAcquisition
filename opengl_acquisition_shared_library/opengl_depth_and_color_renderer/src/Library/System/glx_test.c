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

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../Rendering/ShaderPipeline/shader_loader.h"

#define U 0.5


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


int WIDTH=640;
int HEIGHT=480;


static const float cubeCoords[]=
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

static const float cubeNormals[]={ //X  Y  Z  W
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



// One color for each vertex. They were generated randomly.
static const float cubeColors[] = {
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



float pyramidCoords[]={ //X  Y  Z       W
                     //Far
                     -U, -U, -U,      // bottom left
                      0,  U,  0,      // top
                      U, -U, -U,      // bottom right

                     //Near
                      -U, -U,  U,      // top left
                       U, -U,  U,      // top right
                       0,  U,  0,      // top

                     //Left
                      -U, -U, -U,       // bottom left
                      -U, -U,  U,      // top left
                       0,  U,  0,      // top

                     //Right
                       U, -U, -U,      // bottom right
                       0,  U,  0,      // top
                       U, -U,  U,      // top right


                     //Bottom
                     -U, -U,  U, //1.0,  // top left
                     -U, -U, -U, //1.0,  // bottom left
                      U, -U,  U, //1.0,  // top right

                      U, -U,  U,//1.0,   // top right
                     -U, -U, -U, //1.0,  // bottom left
                      U, -U, -U  //, 1.0 // bottom right
                    };



float pyramidNormals[]={ //X  Y  Z  W
                      0.0,0.4472,-0.8944,
                      0.0,0.4472,0.8944,
                      -0.8944,0.4472,0.0,
                      0.8944,0.4472,0.0,
                      0.0,-1.0,0.0,
                      0.0,-1.0,0.0
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
      fprintf(stderr,RED "OpenGL Error (%d) : %s %d \n ", err , file ,line );
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
    //fprintf(stderr,"Pushing %lu vertices (%u bytes) and %u normals (%u bytes) as our object \n"  ,verticesLength/sizeof(float),verticesLength,normalsLength/sizeof(float),normalsLength);
    glBufferData( GL_ARRAY_BUFFER, verticesLength + normalsLength  + colorsLength  ,NULL, GL_STATIC_DRAW );   checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, 0                                      , verticesLength , vertices );   checkOpenGLError(__FILE__, __LINE__);
    glBufferSubData( GL_ARRAY_BUFFER, verticesLength                         , normalsLength  , normals );    checkOpenGLError(__FILE__, __LINE__);

    if ( (colors!=0) && (colorsLength!=0) )
    {
     glBufferSubData( GL_ARRAY_BUFFER, verticesLength + normalsLength , colorsLength , colors );               checkOpenGLError(__FILE__, __LINE__);
    }

    GLuint vPosition = glGetAttribLocation( programID, "vPosition" );               checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vPosition );                                         checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,(GLvoid*) 0);  checkOpenGLError(__FILE__, __LINE__);


    GLuint vNormal = glGetAttribLocation( programID, "vNormal" );                             checkOpenGLError(__FILE__, __LINE__);
    glEnableVertexAttribArray( vNormal );                                                     checkOpenGLError(__FILE__, __LINE__);
    glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,(GLvoid*) verticesLength ); checkOpenGLError(__FILE__, __LINE__);


    if ( (colors!=0) && (colorsLength!=0) )
    {
     GLuint vColor = glGetAttribLocation( programID, "vColor" );
     glEnableVertexAttribArray( vColor );
     glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,(GLvoid*) ( verticesLength + normalsLength ) );
     checkOpenGLError(__FILE__, __LINE__);
    }

  return buffer;
}


void prepareMatrices(
                    struct Matrix4x4OfFloats * projectionMatrix,
                    struct Matrix4x4OfFloats * viewMatrix,
                    struct Matrix4x4OfFloats * viewportMatrix
                    )
{

     int viewport[4]={0};
     float fx = 535.423889;
     float fy = 533.48468;
     float skew = 0.0;
     float cx = (float) WIDTH/2;
     float cy = (float) HEIGHT/2;
     float near = 1.0;
     float far = 255.0;
     buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor(
                                         projectionMatrix->m ,
                                         viewport ,
                                         fx, fy,
                                         skew,
                                         cx,  cy,
                                         WIDTH, HEIGHT,
                                         near,
                                         far
                                         );
     transpose4x4FMatrix(projectionMatrix->m); //We want our own Row Major format..
     //glViewport(viewport[0],viewport[1],viewport[2],viewport[3]); //<--Does this do anything?


     create4x4FScalingMatrix(viewMatrix,-1.0,1.0,1.0);

     glGetViewportMatrix(viewportMatrix->m,viewport[0],viewport[1],viewport[2],viewport[3],near,far);
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















int doDrawing()
{
   fprintf(stderr," doDrawing \n");


	// Create and compile our GLSL program from the shaders
	//struct shaderObject * sho = loadShader("../../../shaders/TransformVertexShader.vertexshader", "../../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../../shaders/simple.vert", "../../../shaders/simple.frag");
    if (sho==0) { fprintf(stderr,"Could not load..\n"); exit(1); }
    GLuint programID = sho->ProgramObject;





     struct Matrix4x4OfFloats projectionMatrix;
     struct Matrix4x4OfFloats viewportMatrix;
     struct Matrix4x4OfFloats viewMatrix;
     prepareMatrices(
                     &projectionMatrix,
                     &viewMatrix,
                     &viewportMatrix
                    );



 	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    unsigned int wireFrame=0;


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


	do{
       fprintf(stderr,".");
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

                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix
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

                  &projectionMatrix,
                  &viewportMatrix,
                  &viewMatrix
                 );

		// Swap buffers
        glx3_endRedraw();
        usleep(10);
	} // Check if the ESC key was pressed or the window was closed
	while( 1 );

	// Cleanup VBO and shader
	//glDeleteBuffers(1, &vertexbuffer);
	//glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &pyramidVAO);
	glDeleteVertexArrays(1, &cubeVAO);
}














int main(int argc,const char **argv)
{
  if (!start_glx3_stuff(
                        WIDTH,
                        HEIGHT,
                        1, //Force viewing window
                        argc,
                        argv
                        )
     ) 
      { fprintf(stderr,"Could not initialize"); return 1;}


  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 0;
   }

   doDrawing();

  stop_glx3_stuff();
 return 0;
}
