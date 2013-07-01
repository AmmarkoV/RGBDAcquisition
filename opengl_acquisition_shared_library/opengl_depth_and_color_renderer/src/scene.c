#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include <stdio.h>


#include "TrajectoryParser/TrajectoryParser.h"
#include "model_loader.h"
#include "scene.h"

struct VirtualStream * scene = 0;
struct Model ** models=0;

float farPlane = 255;
float nearPlane= 1.0;



const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };


float camera_pos_x = 0.0f; float camera_pos_y = 0.0f; float camera_pos_z = 8.0f;
float camera_angle_x = 0.0f; float camera_angle_y = 0.0f; float camera_angle_z = 0.0f;

unsigned int ticks = 0;



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

  glEnable(GL_COLOR);
  glEnable(GL_COLOR_MATERIAL);

  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
  glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    //glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    //glLightfv(GL_LIGHT0, GL_POSITION, light_position);

  glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);



  float R,G,B,trans;
  scene = createVirtualStream("scene.conf");
  if (scene==0) { fprintf(stderr,"Could not read scene data \n"); return 0; }
  models = (struct Model **) malloc(scene->numberOfObjectTypes * sizeof(struct Model **));

  unsigned int i=0;
  //Object 0 is camera
  for (i=1; i<scene->numberOfObjects; i++)
    {
         models[i] = loadModel("Models/",getObjectTypeModel(scene,i));
         if (models[i]!=0)
          {
           R=1.0f; G=1.0f;  B=0.0f; trans=0.0f;
           getObjectColorsTrans(scene,i,&R,&G,&B,&trans);
           setModelColor(models[i],&R,&G,&B,&trans);
           models[i]->nocolor = scene->object[i].nocolor;
          }
            else
          {
            fprintf(stderr,"Failed loading model %u \n",i);
          }

    }

  return 1;
}


int closeScene()
{
  unsigned int i=0;
  //Object 0 is camera
  for (i=1; i<scene->numberOfObjectTypes; i++)
    {
       unloadModel(models[i]);
    }
  free(models);

  destroyVirtualStream(scene);

  return 1;
}



int tickScene()
{
   // addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll);
   float x,y,z,heading,pitch,roll;
   //addToModelCoordinates(spatoula,0.0 /*X*/,0.0/*Y*/,0.0/*Z*/,(float) 0.01/*HEADING*/,(float) 0.01/*PITCH*/,(float) 0.006/*ROLL*/);

  float posStack[7];
  float * pos = (float*) &posStack;

  unsigned int i=0;
  //Object 0 is camera
  for (i=1; i<scene->numberOfObjects; i++)
    {
       pos[0]=0; pos[1]=0; pos[2]=0; pos[3]=0; pos[4]=0; pos[5]=0; pos[6]=0;
       calculateVirtualStreamPos(scene,i,ticks*100,pos);
       setModelCoordinatesNoSTACK(models[i],&pos[0],&pos[1],&pos[2],&pos[3],&pos[4],&pos[5]);
    }

   //Camera information
   calculateVirtualStreamPos(scene,0,ticks*100,pos);
   camera_pos_x = pos[0];  camera_pos_y = pos[1]; camera_pos_z = pos[2];
   camera_angle_x = pos[3]; camera_angle_y = pos[4]; camera_angle_z = pos[5];

   usleep(100);
   ++ticks;
   return 1;
}



int renderScene()
{
  glEnable (GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  glPushMatrix();
  glMatrixMode(GL_MODELVIEW );
  glLoadIdentity();

  //TODO if calibration is given , change this with the calibration matrix
  //http://www.khronos.org/opengles/sdk/1.1/docs/man/glLoadMatrix.xml

  glRotatef(camera_angle_x,-1.0,0,0); // Peristrofi gyrw apo ton x
  glRotatef(camera_angle_y,0,-1.0,0); // Peristrofi gyrw apo ton y
  glRotatef(camera_angle_z,0,0,-1.0);
  glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);

  unsigned int i;
  //Object 0 is camera
  for (i=1; i<scene->numberOfObjects; i++)
    {
       drawModel(models[i]);
    }


  glPopMatrix();
  return 1;
}

