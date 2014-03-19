#include <stdio.h>
#include <stdlib.h>


#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>
#include <math.h>

float rtri=0.0,rquad=0.0;

int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
   glViewport(0, 0, newWidth,newHeight);
   return 1;
}

void gldPerspective(GLdouble fovx, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
   // This code is based off the MESA source for gluPerspective
   // *NOTE* This assumes GL_PROJECTION is the current matrix

   GLdouble xmin, xmax, ymin, ymax;
   GLdouble m[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   xmax = zNear * tan(fovx * M_PI / 360.0);
   xmin = -xmax;

   ymin = xmin / aspect;
   ymax = xmax / aspect;

   // Set up the projection matrix
   m[0] = (2.0 * zNear) / (xmax - xmin);
   m[5] = (2.0 * zNear) / (ymax - ymin);
   m[10] = -(zFar + zNear) / (zFar - zNear);

   m[8] = (xmax + xmin) / (xmax - xmin);
   m[9] = (ymax + ymin) / (ymax - ymin);
   m[11] = -1.0;

   m[14] = -(2.0 * zFar * zNear) / (zFar - zNear);

   // Add to current matrix
   glMultMatrixd(m);
}


int drawPyramid()
{
  // draw a pyramid (in smooth coloring mode)
  glBegin(GL_POLYGON);				// start drawing a pyramid

  // front face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Set The Color To Red
  glVertex3f(0.0f, 1.0f, 0.0f);		        // Top of triangle (front)
  glColor3f(0.0f,1.0f,0.0f);			// Set The Color To Green
  glVertex3f(-1.0f,-1.0f, 1.0f);		// left of triangle (front)
  glColor3f(0.0f,0.0f,1.0f);			// Set The Color To Blue
  glVertex3f(1.0f,-1.0f, 1.0f);		        // right of traingle (front)

  // right face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Right)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f( 1.0f,-1.0f, 1.0f);		// Left Of Triangle (Right)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f( 1.0f,-1.0f, -1.0f);		// Right Of Triangle (Right)

  // back face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Back)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f( 1.0f,-1.0f, -1.0f);		// Left Of Triangle (Back)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f(-1.0f,-1.0f, -1.0f);		// Right Of Triangle (Back)

  // left face of pyramid.
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Left)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f(-1.0f,-1.0f,-1.0f);		// Left Of Triangle (Left)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f(-1.0f,-1.0f, 1.0f);		// Right Of Triangle (Left)

  glEnd();					// Done Drawing The Pyramid
}

/* The main drawing function. */
void DrawGLScene()
{
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);	// Clear The Screen And The Depth Buffer
  glLoadIdentity();				// Reset The View
glTranslatef(-1.5f,0.0f,-6.0f);		// Move Left 1.5 Units And Into The Screen 6.0

 // drawPyramid();

  glTranslatef(-1.5f,0.0f,-6.0f);		// Move Left 1.5 Units And Into The Screen 6.0

  glRotatef(rtri,0.0f,1.0f,0.0f);		// Rotate The Pyramid On The Y axis

  // draw a pyramid (in smooth coloring mode)
  glBegin(GL_POLYGON);				// start drawing a pyramid

  // front face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Set The Color To Red
  glVertex3f(0.0f, 1.0f, 0.0f);		        // Top of triangle (front)
  glColor3f(0.0f,1.0f,0.0f);			// Set The Color To Green
  glVertex3f(-1.0f,-1.0f, 1.0f);		// left of triangle (front)
  glColor3f(0.0f,0.0f,1.0f);			// Set The Color To Blue
  glVertex3f(1.0f,-1.0f, 1.0f);		        // right of traingle (front)

  // right face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Right)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f( 1.0f,-1.0f, 1.0f);		// Left Of Triangle (Right)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f( 1.0f,-1.0f, -1.0f);		// Right Of Triangle (Right)

  // back face of pyramid
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Back)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f( 1.0f,-1.0f, -1.0f);		// Left Of Triangle (Back)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f(-1.0f,-1.0f, -1.0f);		// Right Of Triangle (Back)

  // left face of pyramid.
  glColor3f(1.0f,0.0f,0.0f);			// Red
  glVertex3f( 0.0f, 1.0f, 0.0f);		// Top Of Triangle (Left)
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f(-1.0f,-1.0f,-1.0f);		// Left Of Triangle (Left)
  glColor3f(0.0f,1.0f,0.0f);			// Green
  glVertex3f(-1.0f,-1.0f, 1.0f);		// Right Of Triangle (Left)

  glEnd();					// Done Drawing The Pyramid

  glLoadIdentity();				// make sure we're no longer rotated.
  glTranslatef(1.5f,0.0f,-7.0f);		// Move Right 3 Units, and back into the screen 7

  glRotatef(rquad,1.0f,1.0f,1.0f);		// Rotate The Cube On X, Y, and Z

  // draw a cube (6 quadrilaterals)
  glBegin(GL_QUADS);				// start drawing the cube.

  // top of cube
  glColor3f(0.0f,1.0f,0.0f);			// Set The Color To Blue
  glVertex3f( 1.0f, 1.0f,-1.0f);		// Top Right Of The Quad (Top)
  glVertex3f(-1.0f, 1.0f,-1.0f);		// Top Left Of The Quad (Top)
  glVertex3f(-1.0f, 1.0f, 1.0f);		// Bottom Left Of The Quad (Top)
  glVertex3f( 1.0f, 1.0f, 1.0f);		// Bottom Right Of The Quad (Top)

  // bottom of cube
  glColor3f(1.0f,0.5f,0.0f);			// Set The Color To Orange
  glVertex3f( 1.0f,-1.0f, 1.0f);		// Top Right Of The Quad (Bottom)
  glVertex3f(-1.0f,-1.0f, 1.0f);		// Top Left Of The Quad (Bottom)
  glVertex3f(-1.0f,-1.0f,-1.0f);		// Bottom Left Of The Quad (Bottom)
  glVertex3f( 1.0f,-1.0f,-1.0f);		// Bottom Right Of The Quad (Bottom)

  // front of cube
  glColor3f(1.0f,0.0f,0.0f);			// Set The Color To Red
  glVertex3f( 1.0f, 1.0f, 1.0f);		// Top Right Of The Quad (Front)
  glVertex3f(-1.0f, 1.0f, 1.0f);		// Top Left Of The Quad (Front)
  glVertex3f(-1.0f,-1.0f, 1.0f);		// Bottom Left Of The Quad (Front)
  glVertex3f( 1.0f,-1.0f, 1.0f);		// Bottom Right Of The Quad (Front)

  // back of cube.
  glColor3f(1.0f,1.0f,0.0f);			// Set The Color To Yellow
  glVertex3f( 1.0f,-1.0f,-1.0f);		// Top Right Of The Quad (Back)
  glVertex3f(-1.0f,-1.0f,-1.0f);		// Top Left Of The Quad (Back)
  glVertex3f(-1.0f, 1.0f,-1.0f);		// Bottom Left Of The Quad (Back)
  glVertex3f( 1.0f, 1.0f,-1.0f);		// Bottom Right Of The Quad (Back)

  // left of cube
  glColor3f(0.0f,0.0f,1.0f);			// Blue
  glVertex3f(-1.0f, 1.0f, 1.0f);		// Top Right Of The Quad (Left)
  glVertex3f(-1.0f, 1.0f,-1.0f);		// Top Left Of The Quad (Left)
  glVertex3f(-1.0f,-1.0f,-1.0f);		// Bottom Left Of The Quad (Left)
  glVertex3f(-1.0f,-1.0f, 1.0f);		// Bottom Right Of The Quad (Left)

  // Right of cube
  glColor3f(1.0f,0.0f,1.0f);			// Set The Color To Violet
  glVertex3f( 1.0f, 1.0f,-1.0f);	        // Top Right Of The Quad (Right)
  glVertex3f( 1.0f, 1.0f, 1.0f);		// Top Left Of The Quad (Right)
  glVertex3f( 1.0f,-1.0f, 1.0f);		// Bottom Left Of The Quad (Right)
  glVertex3f( 1.0f,-1.0f,-1.0f);		// Bottom Right Of The Quad (Right)
  glEnd();					// Done Drawing The Cube

  rtri+=15.0f;					// Increase The Rotation Variable For The Pyramid
  rquad-=15.0f;					// Decrease The Rotation Variable For The Cube

  // swap the buffers to display, since double buffering is used.
//  glutSwapBuffers();

  glx_endRedraw();
}

int main()
{
  float nearPlane = 1.0 , farPlane = 15000 , fieldOfView = 65.0;
  unsigned int width=640,height=480,viewWindow=1;
  char test[12]={0};
  char * testP = test;
  start_glx_stuff(width,height,viewWindow,0,&testP);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);		// This Will Clear The Background Color To Black
  glClearDepth(1.0);				// Enables Clearing Of The Depth Buffer
  glDepthFunc(GL_LESS);			        // The Type Of Depth Test To Do
  glEnable(GL_DEPTH_TEST);		        // Enables Depth Testing
  glShadeModel(GL_SMOOTH);			// Enables Smooth Color Shading

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();				// Reset The Projection Matrix

  //gluPerspective(65.0f,(GLfloat)width/(GLfloat)height,nearPlane,farPlane);	// Calculate The Aspect Ratio Of The Window
  gldPerspective((double) fieldOfView, (double) width/height, (double) nearPlane, (double) farPlane);
  glViewport(0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
    while (1)
    {
      if (glx_checkEvents())
      {
        DrawGLScene();
      }
    }


    return 0;
}
