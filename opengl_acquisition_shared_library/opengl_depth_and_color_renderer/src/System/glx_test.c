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

#define U 0.5
#define BUFFER_OFFSET( offset )   ((GLvoid*) (offset))

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

int doDrawing()
{
   fprintf(stderr," doDrawing \n");

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	GLuint VertexArrayID=0;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
    fprintf(stderr," loadShader \n");
	//struct shaderObject * sho = loadShader("../../shaders/TransformVertexShader.vertexshader", "../../shaders/ColorFragmentShader.fragmentshader");
	struct shaderObject * sho = loadShader("../../shaders/simple.vert", "../../shaders/simple.frag");
    if (sho==0) { fprintf(stderr,"Could not load..\n"); exit(1); }
    GLuint programID = sho->ProgramObject;

    float projectionMatrix[]={ -1.673200,0.000000,0.000000,0.000000,
                                0.000000,2.222853,0.000000,0.000000,
                                0.000000,0.000000,1.007874,1.000000,
                                0.000000,0.000000,-2.007874,0.000000};

    float modelViewMatrix[]={  -10.000000,0.000001,0.000000,0.000000,
                                 0.000000,0.000000,10.000000,0.000000,
                                 0.000001,10.000000,-0.000000,0.000000,
                                 0.192310,0.549760,22.997351,1.000000
                            };

    float MVP[16];

    multiplyTwo4x4FMatrices(MVP,projectionMatrix,modelViewMatrix);
	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");


    float * vertices = cubeCoords;
    float * normals = cubeNormals;
    unsigned int numberOfVertices = sizeof(cubeCoords);
    unsigned int numberOfNormals = sizeof(cubeNormals);


   fprintf(stderr,"Ready to start rendering : ");

    //GLuint vColor = glGetAttribLocation( programID, "vColor" );
    //glEnableVertexAttribArray( vColor );
    //glVertexAttribPointer( vColor, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET( numberOfVertices + numberOfNormals ) );



	do{
        fprintf(stderr,".");
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, MVP);

   fprintf(stderr,"BufferStart");
		glBufferData( GL_ARRAY_BUFFER, numberOfVertices +  numberOfNormals /* + numberOfColors + numberOfTextureCoords */,NULL, GL_STREAM_DRAW );

        glBufferSubData( GL_ARRAY_BUFFER, 0                                      , numberOfVertices , vertices );
        glBufferSubData( GL_ARRAY_BUFFER, numberOfVertices                       , numberOfNormals  , normals );
   fprintf(stderr," ok ");

        GLuint vPosition = glGetAttribLocation( programID, "vPosition" );
        glEnableVertexAttribArray( vPosition );
        glVertexAttribPointer( vPosition, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(0) );
   fprintf(stderr," vPosition ok ");

        GLuint vNormal = glGetAttribLocation( programID, "vNormal" );
        glEnableVertexAttribArray( vNormal );
        glVertexAttribPointer( vNormal, 3, GL_FLOAT, GL_FALSE, 0,BUFFER_OFFSET(numberOfVertices) );
   fprintf(stderr," vNormal ok ");

		// Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, 36*3); // 12*3 indices starting at 0 -> 12 triangles

		//glDisableVertexAttribArray(0);
		//glDisableVertexAttribArray(1);

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
  int WIDTH=640;
  int HEIGHT=480;
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

  while (1)
   {
     glx3_checkEvents();
     fprintf(stderr,".");
     drawGenericTriangleMesh(cubeCoords,cubeNormals,36*3);
     usleep(100);
     glx3_endRedraw();
   }

  stop_glx3_stuff();
 return 0;
}
