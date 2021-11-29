#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <time.h>
#include <math.h>

#include "../Tools/tools.h"

#include "../../../../../tools/AmMatrix/matrixCalculations.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"

#include "../TrajectoryParser/TrajectoryCalculator.h"
#include "../ModelLoader/model_loader.h"
#include "../ModelLoader/model_loader_hardcoded.h"
#include "../ModelLoader/model_loader_tri.h"
#include "../ModelLoader/model_loader_transform_joints.h"

#include "scene.h"

#include "../Rendering/ogl_rendering.h"
#include "../Rendering/tiledRenderer.h"


#include "../OGLRendererSandbox.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

struct VirtualStream * scene = 0;
struct ModelList * modelStorage;

unsigned int WIDTH=640;
unsigned int HEIGHT=480;


const GLfloat light_ambient[]  = { 0.2f, 0.2f, 0.2f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
const GLfloat mat_shininess[]  = { 5.0f };

struct VirtualStream *  getLoadedScene()
{
    return scene;
}

struct ModelList *  getLoadedModelStorage()
{
    return modelStorage;
}


float sceneGetNearPlane()
{
 return scene->controls.nearPlane;
}

float sceneGetDepthScalingPrameter()
{
  return scene->controls.scaleDepthTo;
}



int sceneSetNearFarPlanes(float near, float far)
{
  if (scene==0) { return 0; }
  scene->controls.nearPlane=near;
  scene->controls.farPlane=far;
  return 1;
}

int sceneSeekTime(unsigned int seekTime)
{
  if (scene==0) { return 0; }
  scene->ticks=seekTime;

  return 1;
}

int sceneIgnoreTime(unsigned int newSettingMode)
{
  if (scene==0) { return 0; }
  if (newSettingMode)   { scene->ignoreTime = 1; scene->controls.tickUSleepTime=0;   scene->rate=1.0;  } else
                        { scene->ignoreTime = 0; scene->controls.tickUSleepTime=100; scene->rate=1.0; }

  return 1;
}

int sceneSwitchKeyboardControl(int newVal)
{
  if (scene!=0)
  {
   scene->userCanMoveCameraOnHisOwn=newVal;
  }
 return 1;
}

int sceneSetOpenGLExtrinsicCalibration(struct VirtualStream * scene,float * rodriguez,float * translation ,float scaleToDepthUnit)
{
  if (scene==0) { fprintf(stderr,"Cannot access virtual stream to sceneSetOpenGLExtrinsicCalibration\n"); return 0; }

  scene->useCustomModelViewMatrix=1;
  convertRodriguezAndTranslationToOpenGL4x4ProjectionMatrix(scene->customModelViewMatrix , rodriguez , translation , scaleToDepthUnit);

  scene->controls.scaleDepthTo=(float) scaleToDepthUnit;


  scene->extrinsicsDeclared = 1;
  scene->extrinsicTranslation[0] = translation[0];
  scene->extrinsicTranslation[1] = translation[1];
  scene->extrinsicTranslation[2] = translation[2];

  scene->extrinsicRodriguezRotation[0] = rodriguez[0];
  scene->extrinsicRodriguezRotation[1] = rodriguez[1];
  scene->extrinsicRodriguezRotation[2] = rodriguez[2];

  return 1;
}

int sceneSetOpenGLIntrinsicCalibration(struct VirtualStream * scene,float * camera)
{
  if (scene==0) { fprintf(stderr,"Cannot sceneSetOpenGLIntrinsicCalibration without an initialized VirtualStream\n"); return 0;}

  scene->useIntrinsicMatrix=1;
  scene->cameraMatrix[0]=camera[0];
  scene->cameraMatrix[1]=camera[1];
  scene->cameraMatrix[2]=camera[2];
  scene->cameraMatrix[3]=camera[3];
  scene->cameraMatrix[4]=camera[4];
  scene->cameraMatrix[5]=camera[5];
  scene->cameraMatrix[6]=camera[6];
  scene->cameraMatrix[7]=camera[7];
  scene->cameraMatrix[8]=camera[8];

  updateProjectionMatrix();
  return 1;
}



int sceneSetOpenGLIntrinsicCalibrationNew(struct VirtualStream * scene,float fx,float fy,float cx,float cy,float width,float height,float nearPlane,float farPlane)
{
  if (scene==0) { fprintf(stderr,"Cannot sceneSetOpenGLIntrinsicCalibration without an initialized VirtualStream\n"); return 0;}

  scene->useIntrinsicMatrix=1;
  scene->emulateProjectionMatrix[0]=fx;
  scene->emulateProjectionMatrix[4]=fy;
  scene->emulateProjectionMatrix[2]=cx;
  scene->emulateProjectionMatrix[5]=cy;
  WIDTH = width;
  HEIGHT = height;
  
  scene->controls.nearPlane = nearPlane;
  scene->controls.farPlane  = farPlane;
  scene->emulateProjectionMatrixDeclared = 1;
  scene->useIntrinsicMatrix = 0;
  updateProjectionMatrix();
  return 1;
}


int updateProjectionMatrix()
{
  fprintf(stderr,"updateProjectionMatrix called ( %d x %d )  \n",WIDTH,HEIGHT);
  if (scene==0) { fprintf(stderr,"No Scene declared yet , don't know how to update proj matrix\n"); return 0; }
  if ( scene->emulateProjectionMatrixDeclared)
  {
     if (scene->useIntrinsicMatrix)
     {
       fprintf(stderr,YELLOW "Please note that intrinsics have been passed as an argument but we also have a projection matrix from trajectory parser , we will use the latter\n" NORMAL);
     }
     fprintf(stderr,"Emulating Projection Matrix from Trajectory Parser\n");
     int viewport[4]={0};
     float fx = scene->emulateProjectionMatrix[0];
     float fy = scene->emulateProjectionMatrix[4];
     float skew = 0.0;
     float cx = scene->emulateProjectionMatrix[2];
     float cy = scene->emulateProjectionMatrix[5];
     buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor(scene->projectionMatrix,viewport,fx,fy,skew,cx,cy,WIDTH,HEIGHT,scene->controls.nearPlane,scene->controls.farPlane);
     scene->projectionMatrixDeclared =1;
     fprintf(stderr,"Updated projection matrix using 3x3 matrix\n");
  }

  if ( scene->projectionMatrixDeclared )
  { //Scene configuration overwrites local configuration
    fprintf(stderr,"Custom projection matrix is declared\n");
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf( scene->projectionMatrix ); // we load a matrix of Doubles

    if ( (WIDTH==0) || (HEIGHT==0) ) { fprintf(stderr,"Null dimensions for viewport\n"); }
    glViewport(0,0,WIDTH,HEIGHT);

    print4x4FMatrix("OpenGL Projection Matrix Given by Trajectory Parser\n", scene->projectionMatrix ,0 );

  } else
  if (scene->useIntrinsicMatrix)
  {
   int viewport[4]={0};

   fprintf(stderr,"Using intrinsics to build projection matrix\n");
   buildOpenGLProjectionForIntrinsics_OpenGLColumnMajor(
                                             scene->customProjectionMatrix  ,
                                             viewport ,
                                             scene->cameraMatrix[0],
                                             scene->cameraMatrix[4],
                                             0.0,
                                             scene->cameraMatrix[2],
                                             scene->cameraMatrix[5],
                                             WIDTH,
                                             HEIGHT,
                                             scene->controls.nearPlane ,
                                             scene->controls.farPlane
                                           );

   print4x4FMatrix("OpenGL Projection Matrix", scene->customProjectionMatrix , 0 ); 
   glMatrixMode(GL_PROJECTION);
   glLoadMatrixf(scene->customProjectionMatrix); // we load a matrix of Doubles
   glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
  }
    else
  {
   fprintf(stderr,"Regular Clean/Default Projection matrix \n");
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   float matrix[16]={0}; 
   gldPerspective(
                  matrix,
                  (float) scene->controls.fieldOfView,
                  (float) WIDTH/HEIGHT,
                  (float) scene->controls.nearPlane,
                  (float) scene->controls.farPlane
                 );
   glMultMatrixf(matrix);

   //glFrustum(-1.0, 1.0, -1.0, 1.0, nearPlane , farPlane);
   glViewport(0, 0, WIDTH, HEIGHT);
  }

  fprintf(stderr,"updateProjectionMatrix ( %u x %u ) done\n",WIDTH,HEIGHT);
  return 1;
}

int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
   fprintf(stderr,"Window size changed to %u x %u\n",newWidth,newHeight);
   WIDTH=newWidth;
   HEIGHT=newHeight;
   updateProjectionMatrix();
   return 1;
}

int initScene(int argc,const char *argv[],const char * confFile)
{
  fprintf(stderr,"Initializing Scene\n");

  //Reset renderer options..
  resetRendererOptions();

  //Making enough space for a "handfull" of objects , this has to be allocated before creating the virtual stream to accomodate the 3D models
  modelStorage = allocateModelList(256);

  scene = createVirtualStream(confFile,modelStorage);
  fprintf(stderr,"createVirtualStream returned \n");
  if (scene==0) { fprintf(stderr,RED "Could not read scene data \n" NORMAL); return 0; }
  scene->modelStorage = (void*) modelStorage;


  //This only enables keyfov if enabled in scene
  if (scene->userCanMoveCameraOnHisOwn) { scene->controls.userKeyFOVEnabled=1; }



  #if USE_AMMARSERVER
  if (!initializeWebInterface(argc,argv,scene))
  {
       fprintf(stderr,"Web interface failed..\n");
  }
  #endif // USE_AMMARSERVER



  startOGLRendering();


  fprintf(stderr,YELLOW "\nFinal step , we are done parsing %s ..\n" NORMAL , confFile);

  fprintf(stderr,YELLOW "Which results in the following model state..\n\n" NORMAL );
  printModelList(modelStorage);

  return 1;
}

int closeScene()
{
  stopOGLRendering();

  deallocateModelList(modelStorage);

  unsigned int i=0;
  //Object 0 is camera
  for (i=1; i<scene->numberOfObjectTypes; i++)
    {
//       unloadModel(models[i]);
    }
  //free(models);

  destroyVirtualStream(scene);

  return 1;
}



int nanoTickSleep(long nanoseconds)
{
   struct timespec req, rem;

   req.tv_sec = 0;              
   req.tv_nsec = nanoseconds; 

   return nanosleep(&req , &rem);
}



int tickScene(unsigned int framerate)
{
   if (scene->controls.pauseTicking)
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
  unsigned int timestampToUse = 0;


   if (scene->ignoreTime)
   {
     timestampToUse = scene->ticks;
     ++scene->ticks;
     scene->ticksF = (float) scene->ticks;
     scene->controls.lastTickMillisecond =   scene->ticks;
   } else
  //There are two ways to render, the first is regardless of GPU we try to enforce a specific framerate..
  //The rate is controled using the RATE(x) script command. If we get a rate of 100 it means that we will
  //try to playback a stream captured at 100 frames per second..!
    if(scene->forceRateRegardlessOfGPUSpeed)
    {
        unsigned int thisTickMillisecond = GetTickCountMilliseconds();
        unsigned int millisecondDiff=thisTickMillisecond - scene->controls.lastTickMillisecond;
        float ticksPerMillisecond = (float) scene->rate / 1000;

        //Even if we have incredibly slow or very fast animations we want to be able to progress..
        scene->ticksF += ticksPerMillisecond * millisecondDiff;

        scene->ticks = (unsigned int) scene->ticksF;
        scene->controls.lastTickMillisecond = thisTickMillisecond ;

        timestampToUse = scene->ticks;

        //We have managed to use the correct timestamp regardless of framerate,
        //But if our framerate is larger than the target rate let's sleep a little to
        //conserve our GPU resources..

        //Our Framerate is  scene->controls.lastFramerate
        //Our desired framerate is scene->rate
        if (scene->controls.lastFramerate>scene->rate)
        {
          float framesToSkip = scene->controls.lastFramerate - scene->rate;
          float sleepTime = (float) 1000*1000/framesToSkip;

          //Sleep less time to ensure rendering without intermissions
          sleepTime/=3;

          //fprintf(stderr,"sleepTime:%0.2f\n",sleepTime );
          nanoTickSleep((long) sleepTime);
        }
    } else
    //The second way to render is as fast as possible..!
    //We can manually adjust tickUSleepTime using keyboard if we want..
    {
        timestampToUse = scene->ticks;//*((unsigned int) 100/scene->rate);
        if (framerate>0)
         {
          if (scene->controls.tickUSleepTime>0)
              { nanoTickSleep(scene->controls.tickUSleepTime); }
         }
        ++scene->ticks;
    }

   scene->timestampToUse = timestampToUse;


   calculateVirtualStreamPos(scene,0,scene->timestampToUse,pos,0,&scaleX,&scaleY,&scaleZ);
   scene->cameraPose.posX = scene->cameraUserDelta.posX + pos[0];
   scene->cameraPose.posY = scene->cameraUserDelta.posY + pos[1];
   scene->cameraPose.posZ = scene->cameraUserDelta.posZ + pos[2];

   scene->cameraPose.angleX = scene->cameraUserDelta.angleX + pos[3];
   scene->cameraPose.angleY = scene->cameraUserDelta.angleY + pos[4];
   scene->cameraPose.angleZ = scene->cameraUserDelta.angleZ + pos[5];

   return 1;
}

int getModelAtScreenCoordinates(unsigned int x , unsigned int y)
{
  fprintf(stderr,"getModelAtScreenCoordinates(%u,%u) not implemented\n",x,y);
  return 0;
}

int print3DPoint2DWindowPosition(int objID , float x3D , float y3D , float z3D)
{
      int viewport[4];
      float modelview[16];
      float projection[16];

      float pos3DF[3];
      pos3DF[0]=x3D;
      pos3DF[1]=y3D;
      pos3DF[2]=z3D;

      float win[3]={0};

      glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
      glGetFloatv( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );

      //#warning "All the functions that use gluProject / unproject should be moved in a seperate compartment"
      _glhProjectf( pos3DF , modelview, projection, viewport, win );

      if  (
            (win[0] < 0) || (win[0] >= WIDTH) ||
            (win[1] < 0) || (win[1] >= HEIGHT)
          )
      {
         fprintf(stderr,RED "Warn : Object %u offscreen ( %0.2f , %0.2f , %0.2f ) will end up at %0.2f,%0.2f(%0.2f)\n" NORMAL , objID , x3D , y3D , z3D , win[0],win[1],win[2]);
      }
  return 1;
}

unsigned int *  getObject2DBoundingBoxList(unsigned int * bboxItemsSize)
{
 if (scene==0) { return 0; }
 unsigned int i=0,resIdx=0;

 unsigned int * result = (unsigned int *) malloc(sizeof(unsigned int) * 4 * scene->numberOfObjects );
 *bboxItemsSize=scene->numberOfObjects*4;
 if (result!=0)
 {
  for (i=1; i<scene->numberOfObjects; i++)
    {
     //scene->object[i].model = scene->ticks % scene->object[i].numberOfFrames;
        result[resIdx] =  scene->object[i].bbox2D[0]; ++resIdx;
        result[resIdx] =  scene->object[i].bbox2D[1]; ++resIdx;
        result[resIdx] =  scene->object[i].bbox2D[2]; ++resIdx;
        result[resIdx] =  scene->object[i].bbox2D[3]; ++resIdx;
    }
 }

 return result;
}

int doAllEventTriggers(unsigned int timestampToUse)
{
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
 return 1;
}

int drawAllConnectors(struct VirtualStream * scene,unsigned int timestampToUse, float scaleX, float scaleY, float scaleZ)
{
  //Draw all connectors
  float posStackA[7]={0};
  float posStackB[7]={0};
  float * pos1 = (float*) &posStackA;
  float * pos2 = (float*) &posStackB;


  unsigned int i=0;
  for (i=0; i<scene->numberOfConnectors; i++)
  {
    if (
         ( calculateVirtualStreamPos(scene,scene->connector[i].objID_A,timestampToUse,pos1,0,&scaleX,&scaleY,&scaleZ) ) &&
         ( calculateVirtualStreamPos(scene,scene->connector[i].objID_B,timestampToUse,pos2,0,&scaleX,&scaleY,&scaleZ) )
        )
       {
        /*
        fprintf(stderr,"Draw drawConnector %u( Object %u ( %f %f %f ) to Object %u ( %f %f %f )  )\n",i,
                       scene->connector[i].objID_A , pos1[0],pos1[1],pos1[2],
                       scene->connector[i].objID_B , pos2[0],pos2[1],pos2[2]);*/
        float scale = (float) scene->connector[i].scale;

        drawConnector(
                      pos1,
                      pos2,
                      &scale ,
                      &scene->connector[i].R ,
                      &scene->connector[i].G ,
                      &scene->connector[i].B ,
                      &scene->connector[i].Transparency
                     );
       } else
       {
         fprintf(stderr,YELLOW "Could not determine position of objects for connector %u\n" NORMAL,i);
       }
  }

 return 1;
}

int drawAllSceneObjectsAtPositionsFromTrajectoryParser(struct VirtualStream * scene)
{
 if (scene==0) { return 0; }
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before calling drawAllObjectsAtPositionsFromTrajectoryParser\n"); }



 //unsigned int timestampToUse = scene->ticks*((unsigned int) 100/scene->rate);
 unsigned int timestampToUse = scene->timestampToUse;


  unsigned int i;
  for (i=1; i<scene->numberOfObjects; i++)
    {
     if (scene->object[i].numberOfFrames<=1)
     {
       scene->object[i].lastFrame = 0;
     } else
     {
       scene->object[i].lastFrame = scene->ticks % scene->object[i].numberOfFrames;
     }
    }


  doAllEventTriggers(timestampToUse);





  //This is actually the only visible console output..
  if (scene->ticks%10==0)
  {
    fprintf(stderr,"\r%0.2f FPS - @ %0.2f sec ( %u ticks * %u microseconds [ rate %0.2f ] ) \r",
            scene->controls.lastFramerate,
            (float) timestampToUse/1000,
            scene->ticks,
            scene->controls.tickUSleepTime,
            scene->rate
            );
  }
  //---------------------------------------------------


/*
  int viewport[4];
  float modelview[16];
  float projection[16];
  glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
  glGetFloatv( GL_PROJECTION_MATRIX, projection );
  glGetIntegerv( GL_VIEWPORT, viewport );
  print4x4FMatrix("Projection",projection);
  print4x4FMatrix("ModelView",modelview);*/

  int enableTransformedRendering=1;

  unsigned char noColor=0;
  float posStackA[7]={0};
  float posStackB[7]={0};
  float scaleX=1.0,scaleY=1.0,scaleZ=1.0;
  float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;

  float * joints=0;



  //Object 0 is camera , so we draw object 1 To numberOfObjects-1
  for (i=1; i<scene->numberOfObjects; i++)
    {
       unsigned int objectType_WhichModelToDraw = scene->objectTypes[scene->object[i].type].modelListArrayNumber;
       unsigned int numberOfBones = scene->objectTypes[scene->object[i].type].numberOfBones;

       if (objectType_WhichModelToDraw<modelStorage->currentNumberOfModels)
       {
         struct Model * mod = &modelStorage->models[objectType_WhichModelToDraw];

         //fprintf(stderr,"Drawing model %u/%u ( %s ) \n",objectType_WhichModelToDraw ,modelStorage->currentNumberOfModels,mod->pathOfModel);
         float * pos = (float*) &posStackA;

         if (numberOfBones>0) {
                                //The 4x4 Matrix per joint
                                joints=(float *) malloc(sizeof(float) * numberOfBones * 16);


                                //memset(joints,0,sizeof(float) * numberOfBones * 16); //Clear it ..
                                //We initialize Identity Matrices everywhere..
                                unsigned int z;
                                for (z=0; z<(numberOfBones); z++)
                                 {
                                  float * mat = &joints[16*z];

                                  mat[0]=1.0;  mat[1]=0.0;  mat[2]=0.0;  mat[3]=0.0;
                                  mat[4]=0.0;  mat[5]=1.0;  mat[6]=0.0;  mat[7]=0.0;
                                  mat[8]=0.0;  mat[9]=0.0;  mat[10]=1.0; mat[11]=0.0;
                                  mat[12]=0.0; mat[13]=0.0; mat[14]=0.0; mat[15]=1.0;
                                 }
                              }

         if ( calculateVirtualStreamPos(scene,i,timestampToUse,pos,joints,&scaleX,&scaleY,&scaleZ) )
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
           mod->showSkeleton = scene->showSkeleton;
           mod->highlight = ( scene->selectedObject == i );
           //fprintf(stderr,"Model %s is now RGB(%0.2f,%0.2f,%0.2f) , Transparency %0.2f , ColorDisabled %u\n",scene->object[i].name, mod->colorR, mod->colorG, mod->colorB, mod->transparency,mod->nocolor );



           //fprintf(stderr,"Draw OBJ%u(%f %f %f , %f %f %f %f , trans %f )\n",i,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],trans);

           if (scene->debug)
                { print3DPoint2DWindowPosition(i , pos[0],pos[1],pos[2] ); }


           if ( (modelHasASkinTransformation(mod,joints)) && (enableTransformedRendering) )
           {
            //We need to do joint transforms before draw
            //fprintf(stderr,"Doing joint transforms..!\n");
            struct TRI_Model triModelOut={0};
            struct TRI_Model *triModelIn=(struct TRI_Model*) mod->modelInternalData;


            doModelTransform(
                              &triModelOut ,
                              triModelIn ,
                              joints ,
                              numberOfBones * 16 * sizeof(float) , //Each "joints" matrix is 4x4 and consists of floats 
                              1/*Autodetect default matrices for speedup*/ ,
                              1/*Direct setting of matrices*/,
                              1/*Do Transforms, don't just calculate the matrices*/ ,
                              0 /*Default joint convention*/
                            );
            //fprintf(stderr,"TriOUT Indices %u , TriIN Indices %u \n",triModelOut.header.numberOfIndices,triModelIn->header.numberOfIndices);

            mod->modelInternalData = (void*) &triModelOut;
            // - - - -
            //                       pX     pY     pZ      rX     rY    rZ
            if (! drawModelAt(mod,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],mod->rotationOrder) )
              { fprintf(stderr,RED "Could not draw object %u , type %u \n" NORMAL ,i , objectType_WhichModelToDraw  ); }
            // - - - -

            mod->modelInternalData = (void*) triModelIn;
            deallocInternalsOfModelTri(&triModelOut);

           } else
           {
            //Regular <fast> drawing of model
            //                       pX     pY     pZ      rX     rY    rZ
            if (! drawModelAt(mod,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],mod->rotationOrder) )
               { fprintf(stderr,RED "Could not draw object %u , type %u \n" NORMAL ,i , objectType_WhichModelToDraw  ); }
           }




          scene->object[i].bbox2D[0] = mod->bbox2D[0];         scene->object[i].bbox2D[1] = mod->bbox2D[1];
          scene->object[i].bbox2D[2] = mod->bbox2D[2];         scene->object[i].bbox2D[3] = mod->bbox2D[3];
       } else
       { fprintf(stderr,YELLOW "Could not determine position of object %s (%u) , so not drawing it\n" NORMAL,scene->object[i].name,i); }


       if (joints!=0) { free(joints); joints=0; }

       if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after drawing object %u \n",i); }
      } else
      {
       fprintf(stderr,YELLOW "Scene object %u/%u ( %s ) points to unallocated model %u/%u  , cannot draw it\n" NORMAL,i,scene->numberOfObjects,scene->object[i].name , objectType_WhichModelToDraw , modelStorage->currentNumberOfModels  );
      }
    }



    drawAllConnectors(scene , timestampToUse , scaleX, scaleY, scaleZ);


  return 1;
}

int setupSceneCameraBeforeRendering(struct VirtualStream * scene)
{
 int result = 0;

  if ( (scene!=0) && ( scene->modelViewMatrixDeclared ) )
  {
     //Scene configuration overwrites local configuration
     glLoadMatrixf( scene->modelViewMatrix ); // we load a matrix of Doubles

     copy4x4FMatrix(scene->activeModelViewMatrix , scene->modelViewMatrix);
      if (scene->useCustomModelViewMatrix)
         {
           fprintf(stderr,"Please not that the model view matrix has been overwritten by the scene configuration parameter\n");
           print4x4FMatrix("Scene declared modelview matrix", scene->modelViewMatrix , 0);
         }

   checkOpenGLError(__FILE__, __LINE__);
   result = 1;
   return 1;
  }

  if ( (scene!=0) && (scene->useCustomModelViewMatrix) )
  {
    //We load the matrix produced by convertRodriguezAndTranslationToOpenGL4x4DMatrix
    copy4x4FMatrix(scene->activeModelViewMatrix , scene->customModelViewMatrix);
    glLoadMatrixf((const GLfloat*) scene->customModelViewMatrix);

    /* We flip our coordinate system so it comes straight
       glRotatef(90,-1.0,0,0); //TODO FIX THESE
       glScalef(1.0,1.0,-1.0); //These are now taken into account using scene files ( see SCALE_WORLD , MAP_ROTATIONS )
       glRotatef(180,0.0,0,-1.0);
       checkOpenGLError(__FILE__, __LINE__); */
     result = 1;
  } else
  // we create a modelview matrix on the fly by using the camera declared in trajectory parser
  if (scene!=0)
  {
    /* fprintf(stderr,"Using on the fly rotate/translate rot x,y,z ( %0.2f,%0.2f,%0.2f ) trans x,y,z, (  %0.2f,%0.2f,%0.2f ) \n", camera_angle_x,camera_angle_y,camera_angle_z, camera_pos_x,camera_pos_y,camera_pos_z ); */
    
    struct Matrix4x4OfFloats aMVM={0};
    create4x4FCameraModelViewMatrixForRendering(
                                                &aMVM,
                                                //Rotation Component
                                                scene->cameraPose.angleX,
                                                scene->cameraPose.angleY,
                                                scene->cameraPose.angleZ,
                                                //Translation Component
                                                scene->cameraPose.posX,
                                                scene->cameraPose.posY,
                                                scene->cameraPose.posZ
                                               );
                                               
    copy4x4FMatrix(scene->activeModelViewMatrix,aMVM.m); 
    transpose4x4FMatrix(scene->activeModelViewMatrix);
    glLoadMatrixf(scene->activeModelViewMatrix);

    /*
    glLoadIdentity();
    if (camera_angle_x!=0.0)  { glRotatef(camera_angle_x,-1.0,0,0); }// Rotate around x
    if (camera_angle_y!=0.0)  { glRotatef(camera_angle_y,0,-1.0,0); }// Rotate around y
    if (camera_angle_z!=0.0)  { glRotatef(camera_angle_z,0,0,-1.0); }// Rotate around z
    glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);
    checkOpenGLError(__FILE__, __LINE__); */
    result = 1;
  }

  return result;
}

int renderScene()
{
  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before calling renderScene\n"); }
  if (scene!=0) { glClearColor(scene->backgroundR,scene->backgroundG,scene->backgroundB,0.0); } else
                { glClearColor(0.0,0.0,0.0,0.0); }

  glEnable (GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW );

  //Lighting
  if (scene->useLightingSystem)
  {
     renderOGLLight(
                                          scene->lightPosition,
                                          0,
                                          0
                                        );
  }

  setupSceneCameraBeforeRendering(scene);

  drawAllSceneObjectsAtPositionsFromTrajectoryParser(scene);

  if (checkOpenGLError(__FILE__, __LINE__))
    { fprintf(stderr,RED "OpenGL error after done drawing all objects\n" NORMAL ); }

  //---------------------------------------------------------------
  //------------------- Calculate Framerate -----------------------
  //---------------------------------------------------------------
  unsigned long now=GetTickCountMilliseconds();
  unsigned long elapsedTime=now-scene->controls.lastRenderingTime;
  if (elapsedTime==0) { elapsedTime=1; }
  scene->controls.lastFramerate = (float) 1000/(elapsedTime);
  scene->controls.lastRenderingTime = now;
  //---------------------------------------------------------------
  ++scene->controls.framesRendered;
  //---------------------------------------------------------------

 return 1;
}
