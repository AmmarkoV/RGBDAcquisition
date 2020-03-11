#include "scene.h"
#include "control.h"

#include <stdio.h>
#include <stdlib.h>

#include "../TrajectoryParser/TrajectoryCalculator.h"

#include "../OGLRendererSandbox.h"

int printObjectData(unsigned int objectToPrint)
{
    struct VirtualStream * scene = getLoadedScene();

    fprintf(stderr,"==========================================================================\n");
    fprintf(stderr,"==========================================================================\n");
    fprintf(stderr,"==========================================================================\n");
    fprintf(stderr,"Object %u ",objectToPrint);
    if (objectToPrint==0)
        { fprintf(stderr,"( It is really the camera )");
        }  else
        {
         fprintf(stderr,"\n");


         unsigned int frameNumber = scene->ticks;
         printObjectTrajectory(
                               scene,
                               objectToPrint,
                               frameNumber
                              );
        }
    fprintf(stderr,"==========================================================================\n");
    fprintf(stderr,"==========================================================================\n");
    fprintf(stderr,"==========================================================================\n");
  return 1;
}


int moveObject(unsigned objToMove , float X , float Y , float Z)
{
  struct VirtualStream * scene = getLoadedScene();

  if (objToMove==CAMERA_OBJECT)
  {//If we want to move the camera ( which is object 0)
    scene->cameraUserDelta.posX+=X;
    scene->cameraUserDelta.posY+=Y;
    scene->cameraUserDelta.posZ+=Z;
    fprintf(stderr,"Moving camera %0.2f %0.2f %0.2f..!\n",X,Y,Z);
  } else
  {
    unsigned int frameNumber = scene->ticks;
    //if (frameNumber>0) { frameNumber-=1; }
    fprintf(stderr,"Moving obj %u %0.2f %0.2f %0.2f..!\n",objToMove,X,Y,Z);
    //movePositionOfObjectTrajectorySt(scene,objToMove,scene->ticks,X,Y,Z);
    movePositionOfObjectTrajectory(scene,objToMove,frameNumber,&X,&Y,&Z);
  }
 return 1;
}

int rotateObject(unsigned objToMove , float X , float Y , float Z , float angleDegrees)
{
  struct VirtualStream * scene = getLoadedScene();

  if (objToMove==CAMERA_OBJECT)
  {//If we want to rotate the camera ( which is object 0)
    if ( (X==1.0) && (Y==0.0) && (Z==0.0) ) { scene->cameraUserDelta.angleX+=angleDegrees; } else
    if ( (X==0.0) && (Y==1.0) && (Z==0.0) ) { scene->cameraUserDelta.angleY+=angleDegrees; } else
    if ( (X==0.0) && (Y==0.0) && (Z==1.0) ) { scene->cameraUserDelta.angleZ+=angleDegrees; } else
        {
           fprintf(stderr,"Unhandled camera rotation %0.2f %0.2f %0.2f %0.2f..!\n",X,Y,Z,angleDegrees);
           return 0;
        }
  } else
  {
    unsigned int frameNumber = scene->ticks;
    //if (frameNumber>0) { frameNumber-=1; }
    rotatePositionOfObjectTrajectory(scene,objToMove,frameNumber,&X,&Y,&Z,&angleDegrees);
  }
 return 1;
}




int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
  struct VirtualStream * scene = getLoadedScene();
  int result=0;

  switch (key)
    {
        case 0:
             fprintf(stderr,"Mouse key click @ %u,%u\n",x,y);



        break;
        case 'o' :
        case 'O' :
             fprintf(stderr,"Writing \n");
             writeOpenGLColor("color.pnm",0,0,WIDTH,HEIGHT);
             writeOpenGLDepth("depth.pnm",0,0,WIDTH,HEIGHT);
             fprintf(stderr,"Convert to jpg\n");
             result=system("convert color.pnm image.jpg");
             if (result!=0) { fprintf(stderr,"Failed to convert color to image file..\n"); }
             result=system("convert depth.pnm image_depth.png");
             if (result!=0) { fprintf(stderr,"Failed to convert depth to image file..\n"); }
            return 1;
        break;

        case 1 : //SPACE??
        case ' ' :
            if (scene->controls.pauseTicking) { scene->controls.pauseTicking=0; } else { scene->controls.pauseTicking=1; }
            return 1;
        break;

       case 9 : //TAB
             ++scene->selectedObject;
             scene->selectedObject = scene->selectedObject % scene->numberOfObjects;
            return 1;
       break;

       case 10 : // NOTHING! :P
             if (scene->selectedObject>0) { --scene->selectedObject; } else
                                          { scene->selectedObject = scene->numberOfObjects-1; }
            return 1;
       break;


       case -66 : //F1
            if (scene->controls.tickUSleepTime<=10) { scene->controls.tickUSleepTime=0; } else
                                                    { scene->controls.tickUSleepTime-=10; }
             fprintf(stderr,"tickUSleepTime is now %u \n",scene->controls.tickUSleepTime);
            return 1;
       break;
       case -65 : //F2
            scene->controls.tickUSleepTime+=10;
            fprintf(stderr,"tickUSleepTime is now %u \n",scene->controls.tickUSleepTime);
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
            if (scene->controls.userKeyFOVEnabled==0) { scene->controls.userKeyFOVEnabled=1; } else
                                                      { scene->controls.userKeyFOVEnabled=0; }
            return 1;
       break;
       case -60 : //F7 show last frame
            if (scene->alwaysShowLastFrame==0) { scene->alwaysShowLastFrame=1; } else
                                               { scene->alwaysShowLastFrame=0; }
            return 1;
       break;

       case -59 : //F8 show print object configuration
             printObjectData(scene->selectedObject);
            return 1;
       break;


       case -57: //F10 Dump to file
               writeVirtualStream(scene,"dump.scene");
               saveSnapshotOfObjects();
             return 1;

       case 'I': //Show model internal
       case 'i':
            if (scene->showSkeleton==0) { scene->showSkeleton=1; } else
                                        { scene->showSkeleton=0; }
             return 1;
       break;

    };

    if (!scene->controls.userKeyFOVEnabled) { fprintf(stderr,"User FOV change by keyboard input (%d) is disabled [ add MOVE_VIEW(1) to scene ]\n",(signed int) key); return 0; }
    switch (key)
    {
       case 1 : scene->cameraUserDelta.angleX+=1.0; break;
       case 2 : scene->cameraUserDelta.angleY+=1.0; break;
       case 3 : scene->cameraUserDelta.angleZ+=1.0; break;

       case 'P' :
       case 'p' :
            //Unpause/Pause..
       break;


///// -----------------------------------------------------------------------
       case '[' :
        scene->rate+=10;
       break;
       case ']' :
        scene->rate-=10;
        if (scene->rate<=0.1) { scene->rate=0.1;}
       break;

///// -----------------------------------------------------------------------

       case 'W' :
       case 'w' :
              moveObject(scene->selectedObject,0.0,scene->controls.moveSpeed,0.0);
       break;

       case 'S' :
       case 's' :
              moveObject(scene->selectedObject,0.0,-1*scene->controls.moveSpeed,0.0);
       break;

       case 'A' :
       case 'a' :
              moveObject(scene->selectedObject,scene->controls.moveSpeed,0.0,0.0);
       break;

       case 'D' :
       case 'd' :
              moveObject(scene->selectedObject,-1*scene->controls.moveSpeed,0.0,0.0);
       break;

       case 'Q' :
       case 'q' :
              moveObject(scene->selectedObject,0.0,0.0,scene->controls.moveSpeed);
       break;

       case 'Z' :
       case 'z' :
              moveObject(scene->selectedObject,0.0,0.0,-1*scene->controls.moveSpeed);
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

       default :
        fprintf(stderr,"handleUserInput called for key %c ( %u ) \n",key,(unsigned int) key);
       break;

    }
  return 1;
}
