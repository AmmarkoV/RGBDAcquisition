#include "scene.h"
#include "control.h"

#include <stdio.h>
#include <stdlib.h>


int printObjectData(unsigned int objectToPrint)
{
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
  if (objToMove==0)
  {
    userDeltacamera_pos_x+=X;
    userDeltacamera_pos_y+=Y;
    userDeltacamera_pos_z+=Z;
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
  if (objToMove==0)
  {
    if ( (X==1.0) && (Y==0.0) && (Z==0.0) ) { userDeltacamera_angle_x+=angleDegrees; } else
    if ( (X==0.0) && (Y==1.0) && (Z==0.0) ) { userDeltacamera_angle_y+=angleDegrees; } else
    if ( (X==0.0) && (Y==0.0) && (Z==1.0) ) { userDeltacamera_angle_z+=angleDegrees; } else
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
    switch (key)
    {
        case 'o' :
        case 'O' :
             fprintf(stderr,"Writing \n");
             writeOpenGLColor("color.pnm",0,0,WIDTH,HEIGHT);
             writeOpenGLDepth("depth.pnm",0,0,WIDTH,HEIGHT);
            return 1;
        break;

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

       case 10 : // NOTHING! :P
             if (scene->selectedObject>0) { --scene->selectedObject; } else
                                          { scene->selectedObject = scene->numberOfObjects-1; }
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

    if (!userKeyFOVEnabled) { fprintf(stderr,"User FOV change by keyboard input (%d) is disabled [ add MOVE_VIEW(1) to scene ]\n",(signed int) key); return 0; }
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
              moveObject(scene->selectedObject,0.0,moveSpeed,0.0);
       break;

       case 'S' :
       case 's' :
              moveObject(scene->selectedObject,0.0,-1*moveSpeed,0.0);
       break;

       case 'A' :
       case 'a' :
              moveObject(scene->selectedObject,moveSpeed,0.0,0.0);
       break;

       case 'D' :
       case 'd' :
              moveObject(scene->selectedObject,-1*moveSpeed,0.0,0.0);
       break;

       case 'Q' :
       case 'q' :
              moveObject(scene->selectedObject,0.0,0.0,moveSpeed);
       break;

       case 'Z' :
       case 'z' :
              moveObject(scene->selectedObject,0.0,0.0,-1*moveSpeed);
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
        fprintf(stderr,"handleUserInput called for key %c ( %u ) \n",key,key);
       break;

    }
  return 1;
}


