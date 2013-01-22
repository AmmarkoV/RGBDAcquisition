#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include <stdio.h>

#include "scene.h"




float farPlane = 259;
float nearPlane= 1.0;

GLfloat    xAngle = 42.0, yAngle = 82.0, zAngle = 112.0;


const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

float camera_pos_x = 0.0f; float camera_pos_y = -3.0f; float camera_pos_z = 8.0f;
float camera_angle_x = 10.0f; float camera_angle_y = 3.0f; float camera_angle_z = 0.0f;


struct Model * spatoula=0;


int initScene()
{
 glEnable(GL_DEPTH_TEST); /* enable depth buffering */
  glDepthFunc(GL_LESS);    /* pedantic, GL_LESS is the default */
  glClearDepth(1.0);       /* pedantic, 1.0 is the default */

     glEnable(GL_NORMALIZE);
     glShadeModel(GL_SMOOTH);
     glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
     glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
     glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
     glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

  /* frame buffer clears should be to black */
  glClearColor(0.0, 0.0, 0.0, 0.0);

  /* set up projection transform */
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(-1.0, 1.0, -1.0, 1.0, nearPlane , farPlane);
  /* establish initial viewport */
  /* pedantic, full window size is default viewport */
  glViewport(0, 0, WIDTH, HEIGHT);


  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
  glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    //glLightfv(GL_LIGHT0, GL_POSITION, light_position);

  glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);



  spatoula = loadModel("spatoula.obj");
  setModelColor(spatoula,0.0,1.0,1.0);
}


int closeScene()
{
  unloadModel(spatoula);
}



int tickScene()
{
   // addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
   addToModelCoordinates(spatoula,0.0 /*X*/,0.0/*Y*/,0.0/*Z*/,(float) 0.01/*HEADING*/,(float) 0.01/*PITCH*/,(float) 0.006/*ROLL*/);
   //fprintf(stderr,".");
   usleep(20000);
}



int renderScene()
{
  glPushMatrix();
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glRotatef(camera_angle_x,-1.0,0,0); // Peristrofi gyrw apo ton x
  glRotatef(camera_angle_y,0,-1.0,0); // Peristrofi gyrw apo ton y
  glRotatef(camera_angle_z,0,0,-1.0);
  glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);

      drawModel(spatoula);

  glPopMatrix();
}

