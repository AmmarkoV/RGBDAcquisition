#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <math.h>

#include "tiledRenderer.h"
#include "tools.h"

#include "../../../tools/AmMatrix/matrixCalculations.h"
#include "TrajectoryParser/TrajectoryParser.h"
#include "ModelLoader/model_loader.h"
#include "scene.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


//Shader specific stuff ----------------
char fragmentShaderFile[MAX_FILENAMES]={0};
char * selectedFragmentShader = 0;
char vertexShaderFile[MAX_FILENAMES]={0};
char * selectedVertexShader = 0;
struct shaderObject * loadedShader=0;
//--------------------------------------

struct VirtualStream * scene = 0;
struct Model ** models=0;

unsigned int tickUSleepTime=100;
unsigned int pauseTicking=0;
float farPlane = 255; //<--be aware that this has an effect on the depth maps generated
float nearPlane= 1; //<--this also
float fieldOfView = 65;
float scaleDepthTo =1000.0;

//float depthUnit = 1.0;

unsigned int userKeyFOVEnabled=0;

int WIDTH=640;
int HEIGHT=480;

int framesRendered =0 ;



int useIntrinsicMatrix=0;
double cameraMatrix[9]={
                        0.0 , 0.0 , 0.0 ,
                        0.0 , 0.0 , 0.0 ,
                        0.0 , 0.0 , 0.0
                       };


double customProjectionMatrix[16]={0};




int useCustomModelViewMatrix=0;
double customModelViewMatrix[16]={
                                   1.0 , 0.0 , 0.0 , 0.0 ,
                                   0.0 , 1.0 , 0.0 , 0.0 ,
                                   0.0 , 0.0 , 1.0 , 0.0 ,
                                   0.0 , 0.0 , 0.0 , 1.0
                                 };
double customTranslation[3]={0};
double customRodriguezRotation[3]={0};


#define USE_LIGHTS 1
const GLfloat light_ambient[]  = { 0.2f, 0.2f, 0.2f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
const GLfloat mat_shininess[]  = { 5.0f };


float camera_pos_x = 0.0f; float camera_pos_y = 0.0f; float camera_pos_z = 8.0f;
float camera_angle_x = 0.0f; float camera_angle_y = 0.0f; float camera_angle_z = 0.0f;


float userDeltacamera_pos_x = 0.0f; float userDeltacamera_pos_y = 0.0f; float userDeltacamera_pos_z = 0.0f;
float userDeltacamera_angle_x = 0.0f; float userDeltacamera_angle_y = 0.0f; float userDeltacamera_angle_z = 0.0f;

unsigned int ticks = 0;

unsigned int selectedOBJ=1;




//matrix will receive the calculated perspective matrix.
//You would have to upload to your shader
// or use glLoadMatrixf if you aren't using shaders.

void glhFrustumf2(float *matrix, float left, float right, float bottom, float top,
                  float znear, float zfar)
{
    float temp, temp2, temp3, temp4;
    temp = 2.0 * znear;
    temp2 = right - left;
    temp3 = top - bottom;
    temp4 = zfar - znear;
    matrix[0] = temp / temp2;
    matrix[1] = 0.0;
    matrix[2] = 0.0;
    matrix[3] = 0.0;
    matrix[4] = 0.0;
    matrix[5] = temp / temp3;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = (right + left) / temp2;
    matrix[9] = (top + bottom) / temp3;
    matrix[10] = (-zfar - znear) / temp4;
    matrix[11] = -1.0;
    matrix[12] = 0.0;
    matrix[13] = 0.0;
    matrix[14] = (-temp * zfar) / temp4;
    matrix[15] = 0.0;
}


void glhPerspectivef2(float *matrix, float fovyInDegrees, float aspectRatioV,
                      float znear, float zfar)
{
    float ymax, xmax;
    ymax = znear * tan(fovyInDegrees * M_PI / 360.0);
    //ymin = -ymax;
    //xmin = -ymax * aspectRatioV;
    xmax = ymax * aspectRatioV;
    glhFrustumf2(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
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




int updateProjectionMatrix()
{
  fprintf(stderr,"updateProjectionMatrix called ( %u x %u )  \n",WIDTH,HEIGHT);
  if (scene==0) { fprintf(stderr,"No Scene declared yet , don't know how to update proj matrix\n"); return 0; }
  if ( scene->emulateProjectionMatrixDeclared)
  {
     if (useIntrinsicMatrix)
     {
       fprintf(stderr,YELLOW "Please note that intrinsics have been passed as an argument but we also have a projection matrix from trajectory parser , we will use the latter\n" NORMAL);
     }
     fprintf(stderr,"Emulating Projection Matrix from Trajectory Parser");
     int viewport[4]={0};
     double fx = scene->emulateProjectionMatrix[0];
     double fy = scene->emulateProjectionMatrix[4];
     double skew = 0.0;
     double cx = scene->emulateProjectionMatrix[2];
     double cy = scene->emulateProjectionMatrix[5];
     buildOpenGLProjectionForIntrinsics( scene->projectionMatrix , viewport , fx, fy, skew, cx,  cy, WIDTH, HEIGHT, nearPlane, farPlane);
     scene->projectionMatrixDeclared =1;
     fprintf(stderr,"Updated projection matrix using 3x3 matrix");
  }

  if ( scene->projectionMatrixDeclared )
  { //Scene configuration overwrites local configuration
    fprintf(stderr,"Custom projection matrix is declared\n");
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd( scene->projectionMatrix ); // we load a matrix of Doubles
    glViewport(0,0,WIDTH,HEIGHT);

    print4x4DMatrix("OpenGL Projection Matrix Given by Trajectory Parser", scene->projectionMatrix );

  } else
  if (useIntrinsicMatrix)
  {
   int viewport[4]={0};

   fprintf(stderr,"Using intrinsics to build projection matrix\n");
   buildOpenGLProjectionForIntrinsics   (
                                             customProjectionMatrix  ,
                                             viewport ,
                                             cameraMatrix[0],
                                             cameraMatrix[4],
                                             0.0,
                                             cameraMatrix[2],
                                             cameraMatrix[5],
                                             WIDTH,
                                             HEIGHT,
                                             nearPlane ,
                                             farPlane
                                           );

   print4x4DMatrix("OpenGL Projection Matrix", customProjectionMatrix );

   glMatrixMode(GL_PROJECTION);
   glLoadMatrixd(customProjectionMatrix); // we load a matrix of Doubles
   glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
  }
    else
  {
   fprintf(stderr,"Regular Clean/Default Projection matrix \n");
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gldPerspective((double) fieldOfView, (double) WIDTH/HEIGHT, (double) nearPlane, (double) farPlane);
   //glFrustum(-1.0, 1.0, -1.0, 1.0, nearPlane , farPlane);
   glViewport(0, 0, WIDTH, HEIGHT);
  }

  return 1;
}



int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
   WIDTH=newWidth;
   HEIGHT=newHeight;
   updateProjectionMatrix();
   return 1;
}

int moveObject(unsigned objToMove , float X , float Y , float Z)
{
  if (objToMove==0)
  {
    userDeltacamera_pos_x+=X;
    userDeltacamera_pos_y+=Y;
    userDeltacamera_pos_z+=Z;
    fprintf(stderr,"Moving camera %0.2f %0.2f %0.2f..!\n",X,Y,Z);
  } else
  {
    fprintf(stderr,"Moving an arbitrary object is not yet implemented..!\n");
  }
}

int rotateObject(unsigned objToMove , float X , float Y , float Z , float angleDegrees)
{
  if (objToMove==0)
  {
    if ( (X==1.0) && (Y==0.0) && (Z==0.0) ) { userDeltacamera_angle_x+=angleDegrees; } else
    if ( (X==0.0) && (Y==1.0) && (Z==0.0) ) { userDeltacamera_angle_y+=angleDegrees; } else
    if ( (X==0.0) && (Y==0.0) && (Z==1.0) ) { userDeltacamera_angle_z+=angleDegrees; } else
        {
           fprintf(stderr,"Unhandled camera rotation %0.2f %0.2f %0.2f %0.2f..!\n",X,Y,Z,angleDegrees);
        }
  } else
  {
    fprintf(stderr,"Rotating an arbitrary object is not yet implemented..!\n");
  }
}




int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    switch (key)
    {
        case 1 : //SPACE??
        case ' ' :
            if (pauseTicking) { pauseTicking=0; } else { pauseTicking=1; }
            return 1;
        break;

       case 9 : //TAB
             ++scene->selectedObject;
             scene->selectedObject = scene->selectedObject % scene->numberOfObjects;
            return 1;
       break;


       case -66 : //F1
            if (tickUSleepTime<=10) { tickUSleepTime=0; } else
                                    { tickUSleepTime-=10; }
             fprintf(stderr,"tickUSleepTime is now %u \n",tickUSleepTime);
            return 1;
       break;
       case -65 : //F2
            tickUSleepTime+=10;
            fprintf(stderr,"tickUSleepTime is now %u \n",tickUSleepTime);
            return 1;
       break;
       case -64 : //F3
            if (scene->ignoreTime) { scene->ignoreTime=0; } else { scene->ignoreTime=1; }
            return 1;
       break;
       case -63 : //F4
            if (scene->renderWireframe) { scene->renderWireframe=0; } else { scene->renderWireframe=1; }
            return 1;
       break;
       case -62 : //F5 refresh
            if (scene!=0)
             { scene->autoRefreshForce=1; }
            return 1;
       break;
       case -61 : //F6 refresh
            if (userKeyFOVEnabled==0) { userKeyFOVEnabled=1; } else
                                      { userKeyFOVEnabled=0; }
            return 1;
       break;

    };

    if (!userKeyFOVEnabled) { fprintf(stderr,"User FOV change by keyboard input (%d) is disabled [ add MOVE_VIEW(1) to scene ]\n",(signed int) key); return 0; }
    fprintf(stderr,"handleUserInput called for key %c ( %u ) \n",key,key);
    switch (key)
    {
       case 1 : userDeltacamera_angle_x+=1.0; break;
       case 2 : userDeltacamera_angle_y+=1.0; break;
       case 3 : userDeltacamera_angle_z+=1.0; break;

       case 'P' :
       case 'p' :
            //Unpause/Pause..
       break;

///// -----------------------------------------------------------------------

       case 'W' :
       case 'w' :
              moveObject(scene->selectedObject,0.0,1.0,0.0);
       break;

       case 'S' :
       case 's' :
              moveObject(scene->selectedObject,0.0,-1.0,0.0);
       break;

       case 'A' :
       case 'a' :
              moveObject(scene->selectedObject,1.0,0.0,0.0);
       break;

       case 'D' :
       case 'd' :
              moveObject(scene->selectedObject,-1.0,0.0,0.0);
       break;

       case 'Q' :
       case 'q' :
              moveObject(scene->selectedObject,0.0,0.0,1.0);
       break;

       case 'Z' :
       case 'z' :
              moveObject(scene->selectedObject,0.0,0.0,-1.0);
       break;


///// -----------------------------------------------------------------------

       case 'T' :
       case 't' :
              rotateObject(scene->selectedObject,1.0,0.0,0.0,1.0);
       break;
       case 'G' :
       case 'g' :
              rotateObject(scene->selectedObject,1.0,0.0,0.0,-1.0);
       break;
       case 'F' :
       case 'f' :
              rotateObject(scene->selectedObject,0.0,1.0,0.0,1.0);
       break;
       case 'H' :
       case 'h' :
              rotateObject(scene->selectedObject,0.0,1.0,0.0,-1.0);
       break;
       case 'R' :
       case 'r' :
              rotateObject(scene->selectedObject,0.0,0.0,1.0,1.0);
       break;
       case 'Y' :
       case 'y' :
              rotateObject(scene->selectedObject,0.0,0.0,1.0,-1.0);
       break;

    }
  return 1;
}




int initScene(char * confFile)
{
  fprintf(stderr,"Initializing Scene\n");
  scene = createVirtualStream(confFile);
  fprintf(stderr,"createVirtualStream returned \n");
  if (scene==0) { fprintf(stderr,RED "Could not read scene data \n" NORMAL); return 0; }

  //This only enables keyfov if enabled in scene
  if (scene->userCanMoveCameraOnHisOwn) { userKeyFOVEnabled=1; }

  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error while initializing scene\n"); }
  glEnable(GL_DEPTH_TEST); /* enable depth buffering */
  glDepthFunc(GL_LESS);    /* pedantic, GL_LESS is the default */
  glDepthMask(GL_TRUE);
  glClearDepth(1.0);       /* pedantic, 1.0 is the default */

  //HQ settings
  glEnable(GL_NORMALIZE);
  glShadeModel(GL_SMOOTH);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error while initializing HQ settings\n"); }

  /* frame buffer clears should be to black */
  glClearColor(0.0, 0.0, 0.0, 0.0);

  /* set up projection transform */
  glMatrixMode(GL_PROJECTION);

  updateProjectionMatrix();
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after updating projection matrix\n"); }

  /* establish initial viewport */
  /* pedantic, full window size is default viewport */


  #warning "GL_COLOR does not even exist"
  //glEnable(GL_COLOR);
  //if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color \n"); }
  glEnable(GL_COLOR_MATERIAL);
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling color material\n"); }

  #if USE_LIGHTS
   glEnable(GL_LIGHT0);
   glEnable(GL_LIGHTING);
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after enabling lighting\n"); }
   glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
   glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
   glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
   glLightfv(GL_LIGHT0, GL_POSITION, light_position);
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up lights\n"); }

   glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,    mat_ambient);
   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,    mat_diffuse);
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,   mat_specular);
   glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS,   mat_shininess); // <- this was glMateriali
   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting up Front/Back lights\n"); }
  #else
   fprintf(stderr,"Please note that lighting is disabled via the USE_LIGHTS precompiler define\n");
  #endif // USE_LIGHTS


  if ( ( selectedFragmentShader != 0) || ( selectedVertexShader != 0 ) )
  {
      loadedShader = loadShader(selectedVertexShader,selectedFragmentShader);
  }

  //This is not needed -> :P  glCullFace(GL_FRONT_AND_BACK);

  models = (struct Model **) malloc(scene->numberOfObjectTypes * sizeof(struct Model **));
  memset(models,0,scene->numberOfObjectTypes * sizeof(struct Model **));

  unsigned int i=0;  //Object 0 is camera so we don't need to load a model or something for it
  for (i=1; i<scene->numberOfObjectTypes; i++)
    {
         fprintf(stderr,"Loading Model %s ( %u )\n",scene->object[i].name,i);

         models[i] = findModel(models,scene->numberOfObjectTypes,"Models/",getObjectTypeModel(scene,i));
         if (models[i]==0)
         { //If we can't find an already loaded version of the mesh we are looking for
           models[i] = loadModel("Models/",getObjectTypeModel(scene,i));

           if (models[i]!=0)
             { fprintf(stderr,GREEN "Model %s , is now loaded as model[%u] \n" NORMAL,getObjectTypeModel(scene,i) ,i ); } else
             { fprintf(stderr,RED "Failed loading model %s ( %u ) \n" NORMAL,getObjectTypeModel(scene,i),i);            }
         }
    }

  return 1;
}


int closeScene()
{

  if ( ( selectedFragmentShader != 0) || ( selectedVertexShader != 0 ) )
  {
      unloadShader(loadedShader);
  }

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
   if (pauseTicking)
   {
       //No Tick
       return 0;
   }

   //ALL positions should be calculated here!
   //i dont like the way this is working now
   float posStack[7]={0};
   float * pos = (float*) &posStack;
   float scaleX = 1.0 , scaleY = 1.0 , scaleZ = 1.0;

  //Object 0 is camera  lets calculate its position
   calculateVirtualStreamPos(scene,0,ticks*tickUSleepTime,pos,&scaleX,&scaleY,&scaleZ);
   camera_pos_x = userDeltacamera_pos_x + pos[0];  camera_pos_y = userDeltacamera_pos_y + pos[1]; camera_pos_z = userDeltacamera_pos_z + pos[2];
   camera_angle_x = userDeltacamera_angle_x + pos[3]; camera_angle_y = userDeltacamera_angle_y + pos[4]; camera_angle_z = userDeltacamera_angle_z + pos[5];

   if (tickUSleepTime>0)
    { usleep(tickUSleepTime); }

   ++ticks;
   return 1;
}





int print3DPoint2DWindowPosition(int objID , float x3D , float y3D , float z3D)
{
      GLint viewport[4];
      GLdouble modelview[16];
      GLdouble projection[16];

      GLdouble posX = x3D , posY = y3D , posZ = z3D;
      GLdouble winX, winY, winZ=0.0;

      glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
      glGetDoublev( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );

      #warning "All the functions that use gluProject / unproject should be moved in a seperate compartment"
      gluProject( posX, posY, posZ , modelview, projection, viewport, &winX, &winY, &winZ);

      if  (
            (winX < 0) || (winX >= WIDTH) ||
            (winY < 0) || (winY >= HEIGHT)
          )
      {
         fprintf(stderr,"Warn : Object %u offscreen ( %0.2f , %0.2f , %0.2f ) will end up at %0.2f,%0.2f(%0.2f)\n" , objID , x3D , y3D , z3D , winX,winY , winZ);
      }
  return 1;
}



int drawAllObjectsAtPositionsFromTrajectoryParser()
{
 if (scene==0) { return 0; }
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before calling drawAllObjectsAtPositionsFromTrajectoryParser\n"); }


 unsigned int timestampToUse = ticks*100;

  unsigned int i;
  for (i=0; i<scene->numberOfEvents; i++)
  {
     unsigned int objID_A = scene->event[i].objID_A , objID_B = scene->event[i].objID_B;

     switch (scene->event[i].eventType)
      {
        case EVENT_INTERSECTION :

          fprintf(stderr,"Testing Rule %u , intersection between %u and %u \n",i,objID_A,objID_B);
          if ( objectsCollide(scene,timestampToUse,objID_A ,objID_B) )
          {
             if (!scene->event[i].activated)
             {
                scene->event[i].activated=1;
                fprintf(stderr,"Executed %s collision between %u and %u , rule %u/%u \n",scene->event[i].data, objID_A ,objID_B,i,scene->numberOfEvents);

                int retres=system(scene->event[i].data);
                if (retres==0) { fprintf(stderr,"Successfully executed\n"); }
             }
          } else
          {
             if (scene->event[i].activated) { scene->event[i].activated=0;  } //Intersection Event has stopped
          }

        break;
      };
  }




  unsigned char noColor=0;
  float posStackA[7]={0};
  float posStackB[7]={0};
  float scaleX=1.0,scaleY=1.0,scaleZ=1.0;
  float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;


  if (ticks%10==0)
  {
    fprintf(stderr,"\rPlayback %0.2f sec ( %u ticks * %u microseconds ) \r",(float) timestampToUse/1000,ticks,tickUSleepTime);
  }
  //Object 0 is camera , so we draw object 1 To numberOfObjects-1
  for (i=1; i<scene->numberOfObjects; i++)
    {
       struct Model * mod = models[scene->object[i].type];
       float * pos = (float*) &posStackA;
       if ( calculateVirtualStreamPos(scene,i,timestampToUse,pos,&scaleX,&scaleY,&scaleZ) )
       {
         //This is a stupid way of passing stuff to be drawn
         R=1.0f; G=1.0f;  B=1.0f; trans=0.0f; noColor=0;
         getObjectColorsTrans(scene,i,&R,&G,&B,&trans,&noColor);
         //fprintf(stderr,"Object %s should be RGB(%0.2f,%0.2f,%0.2f) , Transparency %0.2f , ColorDisabled %u\n",scene->object[i].name,R,G,B,trans,noColor);
         setModelColor(mod,&R,&G,&B,&trans,&noColor);
         mod->scaleX = scaleX;//scene->object[i].scale;
         mod->scaleY = scaleY;//scene->object[i].scale;
         mod->scaleZ = scaleZ;//scene->object[i].scale;
         mod->wireframe = scene->renderWireframe;
         mod->highlight = ( scene->selectedObject == i );
         //fprintf(stderr,"Model %s is now RGB(%0.2f,%0.2f,%0.2f) , Transparency %0.2f , ColorDisabled %u\n",scene->object[i].name, mod->colorR, mod->colorG, mod->colorB, mod->transparency,mod->nocolor );


         //fprintf(stderr,"Draw OBJ%u(%f %f %f , %f %f %f %f , trans %f )\n",i,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],trans);

         if (scene->debug)
                { print3DPoint2DWindowPosition(i , pos[0],pos[1],pos[2] ); }

         if (! drawModelAt(mod,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]) )
             { fprintf(stderr,RED "Could not draw object %u , type %u \n" NORMAL ,i , scene->object[i].type ); }
       } else
       { fprintf(stderr,YELLOW "Could not determine position of object %s (%u) , so not drawing it\n" NORMAL,scene->object[i].name,i); }
       if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after drawing object %u \n",i); }
    }




  float * pos1 = (float*) &posStackA;
  float * pos2 = (float*) &posStackB;

  for (i=0; i<scene->numberOfConnectors; i++)
  {
    if (
        ( calculateVirtualStreamPos(scene,scene->connector[i].objID_A,timestampToUse,pos1,&scaleX,&scaleY,&scaleZ) ) &&
        ( calculateVirtualStreamPos(scene,scene->connector[i].objID_B,timestampToUse,pos2,&scaleX,&scaleY,&scaleZ) )
        )
       {
        /*
        fprintf(stderr,"Draw drawConnector %u( Object %u ( %f %f %f ) to Object %u ( %f %f %f )  )\n",i,
                       scene->connector[i].objID_A , pos1[0],pos1[1],pos1[2],
                       scene->connector[i].objID_B , pos2[0],pos2[1],pos2[2]);*/
        float scale = (float) scene->connector[i].scale;

        drawConnector(pos1,
                      pos2,
                      &scale ,
                      scene->connector[i].R ,
                      scene->connector[i].G ,
                      scene->connector[i].B ,
                      scene->connector[i].Transparency );
       } else
       {
         fprintf(stderr,YELLOW "Could not determine position of objects for connector %u\n" NORMAL,i);
       }
  }


  return 1;
}







int renderScene()
{
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before calling renderScene\n"); }
  if (scene!=0) { glClearColor(scene->backgroundR,scene->backgroundG,scene->backgroundB,0.0); } else
                { glClearColor(0.0,0.0,0.0,0.0); }

  glEnable (GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW );


  if ( (scene!=0) && ( scene->modelViewMatrixDeclared ) )
  { //Scene configuration overwrites local configuration



   glLoadMatrixd( scene->modelViewMatrix ); // we load a matrix of Doubles
      if (useCustomModelViewMatrix)
         {
           fprintf(stderr,"Please not that the model view matrix has been overwritten by the scene configuration parameter\n");
         }

   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting modelview matrix\n"); }
   //print4x4DMatrix("OpenGL ModelView Matrix Given by Trajectory Parser", scene->modelViewMatrix );
  } //else //<- this else
  //If setOpenGLExtrinsicCalibration has set a custom MODELVIEW matrix we will use it
  #warning "Is this else required?"
  if (useCustomModelViewMatrix)
  {
    //We load the matrix produced by convertRodriguezAndTranslationToOpenGL4x4DMatrix
    glLoadMatrixd((const GLdouble*) customModelViewMatrix);
    // We flip our coordinate system so it comes straight

    //glRotatef(90,-1.0,0,0); //TODO FIX THESE
    //glScalef(1.0,1.0,-1.0); //These are now taken into account using scene files ( see SCALE_WORLD , MAP_ROTATIONS )
    //glRotatef(180,0.0,0,-1.0);

   if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting custom modelview matrix\n"); }

  } else
  // we create a modelview matrix on the fly by using the camera declared in trajectory parser
  {
    /*
    fprintf(stderr,"Using on the fly rotate/translate rot x,y,z ( %0.2f,%0.2f,%0.2f ) trans x,y,z, (  %0.2f,%0.2f,%0.2f ) \n",
             camera_angle_x,camera_angle_y,camera_angle_z,
             camera_pos_x,camera_pos_y,camera_pos_z
            );
    */
    glLoadIdentity();

    if (camera_angle_x!=0.0)
      glRotatef(camera_angle_x,-1.0,0,0); // Rotate around x
    if (camera_angle_y!=0.0)
      glRotatef(camera_angle_y,0,-1.0,0); // Rotate around y
    if (camera_angle_z!=0.0)
      glRotatef(camera_angle_z,0,0,-1.0); // Rotate around z

    glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);
    if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after setting specifying camera position\n"); }
  }

  drawAllObjectsAtPositionsFromTrajectoryParser();

  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after drawing all objects\n"); }

  ++framesRendered;

 return 1;
}







int setupPhotoshoot(
                        void * context,
                        int objID,
                        unsigned int columns , unsigned int rows ,
                        float distance,
                        float angleX,float angleY,float angleZ ,
                        float angXVariance ,float angYVariance , float angZVariance
                       )
{

  struct tiledRendererConfiguration * configuration = (struct tiledRendererConfiguration *) context;

  configuration->columns=columns;
  configuration->rows=rows;
  configuration->objID=objID;
  configuration->distance=distance;
  configuration->angleX=angleX;
  configuration->angleY=angleY;
  configuration->angleZ=angleZ;
  configuration->angXVariance=angXVariance;
  configuration->angYVariance=angYVariance;
  configuration->angZVariance=angZVariance;

  configuration->scenePTR = (void *) scene;
  configuration->modelPTR = (void *) models;
  return 1;
}

void * createPhotoshoot(
                        int objID,
                        unsigned int columns , unsigned int rows ,
                        float distance,
                        float angleX,float angleY,float angleZ ,
                        float angXVariance ,float angYVariance , float angZVariance
                       )
{

  struct tiledRendererConfiguration * configuration = 0;

  configuration = (struct tiledRendererConfiguration * ) malloc(sizeof( struct tiledRendererConfiguration));
  if (configuration == 0) { fprintf(stderr,"Could not allocate a configuration structure\n"); return 0; }


  configuration->columns=columns;
  configuration->rows=rows;
  configuration->objID=objID;
  configuration->distance=distance;
  configuration->angleX=angleX;
  configuration->angleY=angleY;
  configuration->angleZ=angleZ;
  configuration->angXVariance=angXVariance;
  configuration->angYVariance=angYVariance;
  configuration->angZVariance=angZVariance;

  configuration->scenePTR = (void *) scene;
  configuration->modelPTR = (void *) models;



  return (void*) configuration;

}



int renderPhotoshoot( void * context  )
{
  struct tiledRendererConfiguration * configuration=context;

  fprintf(stderr," renderPhotoshoot Rows/Cols %u/%u  Distance %0.2f , Angles %0.2f %0.2f %0.2f\n",configuration->rows,configuration->columns,configuration->distance,configuration->angleX,configuration->angleY,configuration->angleZ);
  fprintf(stderr,"Angle Variance %0.2f %0.2f %0.2f\n",configuration->angXVariance,configuration->angYVariance,configuration->angZVariance);


  int i= tiledRenderer_Render(configuration);

  if (i) { framesRendered++; return 1; }
  return 0;
}



int sceneSwitchKeyboardControl(int newVal)
{
  if (scene!=0)
  {
   scene->userCanMoveCameraOnHisOwn=newVal;
  }
 return 1;
}


