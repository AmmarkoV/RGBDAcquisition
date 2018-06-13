#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <GL/gl.h>
#include <GL/glx.h>

#include "glx3.h"


#define U 0.5

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
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
}

int main(int argc, char **argv)
{
  int WIDTH=640;
  int HEIGHT=480;
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);


  glClearColor ( 1, 0.5, 0, 1 );
  glClear ( GL_COLOR_BUFFER_BIT );
  glx3_endRedraw();

  while (1)
   {
     glx3_checkEvents();
     fprintf(stderr,".");
     drawGenericTriangleMesh(cubeCoords,cubeNormals,36*3);
     sleep(1);
     glx3_endRedraw();
   }
  
  stop_glx3_stuff();
 return 0;
}
