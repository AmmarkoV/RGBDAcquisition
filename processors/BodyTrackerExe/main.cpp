
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

//#define ALL_HEADERS_IN_SAME_DIR 1

#include "../../acquisition/Acquisition.h"
#include "../../tools/Calibration/calibration.h"

#include "../BodyTracker/forth_skeleton_tracker_redist/headers/FORTHUpperBodyGestureTrackerLib.h"

#define LINK_GESTURE_TESTER 1

using namespace std;

char inputname[512]={0};
char outputfoldername[512]={0};

unsigned int delay = 0;

int calibrationSet = 0;
struct calibration calib;

unsigned int devID=0;
ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

struct calibrationFUBGT gestureCalibration={0};
struct calibrationFUBGT * activeGestureCalibration = 0;


void testerGesture(struct handGesture * gesture)
{
  int payload=0;
  char command[512]={0};
    switch (gesture->gestureID)
    {
     case GESTURE_NONE   :   break;
     case GESTURE_CANCEL : payload=1; snprintf(command,512,"wget http://127.0.0.1:8080/index.html?RIGHT=1 -o wgetres"); break;
     case GESTURE_HELP   : payload=1; snprintf(command,512,"eject &");  break;
     case GESTURE_YES    : payload=1; snprintf(command,512,"wget http://192.168.1.19:8080/commander.html?activate=a -o wgetres");
                                      snprintf(command,512,"wget http://127.0.0.1:8080/commander.html?activate=a -o wgetres");   break;
     case GESTURE_NO     : payload=1;
                                      snprintf(command,512,"wget http://192.168.1.19:8080/commander.html?deactivate=a -o wgetres");
                                      snprintf(command,512,"wget http://127.0.0.1:8080/commander.html?deactivate=a -o wgetres");
                                      break;
     case GESTURE_REWARD : payload=1;
                                      snprintf(command,512,"wget http://127.0.0.1:8080/index.html?STARS=1 -o wgetres");
                                      snprintf(command,512,"wget http://127.0.0.1:8080/commander.html?deactivate=all -o wgetres");
                                      break;
     case GESTURE_POINT  : break;
     case GESTURE_COME   : break;
     case GESTURE_WAVE   : return; /*disabled*/ break;
     default :              break;
    };


 if (payload)
 {
   int i=system(command);

   if (i==0) { fprintf(stderr,"Success executing `%s`\n",command); } else
             { fprintf(stderr,"Failed executing `%s`\n",command); }
 }

}


void broadcastNewGesture(unsigned int frameNumber,struct handGesture * gesture)
{
    if (gesture==0)  { return ; }

    #if LINK_GESTURE_TESTER
     testerGesture(gesture); // simple tester function for debugging
    #endif

    switch (gesture->gestureID)
    {
     case GESTURE_NONE   : fprintf(stderr,"G_NONE");   break;
     case GESTURE_CANCEL : fprintf(stderr,"G_CANCEL"); break;
     case GESTURE_HELP   : fprintf(stderr,"G_HELP");   break;
     case GESTURE_YES    : fprintf(stderr,"G_YES");    break;
     case GESTURE_NO     : fprintf(stderr,"G_NO");     break;
     case GESTURE_REWARD : fprintf(stderr,"G_REWARD"); break;
     case GESTURE_POINT  : fprintf(stderr,"G_POINT");  break;
     case GESTURE_COME   : fprintf(stderr,"G_COME");  break;
     case GESTURE_WAVE   : fprintf(stderr,"G_WAVE"); return; /*disabled*/ break;
     default :             fprintf(stderr,"G_NOTFOUND"); break;
    };

    return ;
}


void broadcastNewSkeleton(unsigned int frameNumber,unsigned int skeletonID , struct skeletonHuman * skeletonFound )
{
}


int updateCalibration()
{
  struct calibration calib;
  if ( acquisitionGetColorCalibration(moduleID,devID,&calib)  )
       {
         gestureCalibration.width = calib.width;
         gestureCalibration.height = calib.height;

         unsigned int i=0;
         for (i=0; i<9; i++)
          { gestureCalibration.intrinsic[i] = calib.intrinsic[i];  }

         gestureCalibration.k1 = calib.k1;
         gestureCalibration.k2 = calib.k2;
         gestureCalibration.k3 = calib.k3;
         gestureCalibration.p1 = calib.p1;
         gestureCalibration.p2 = calib.p2;

         activeGestureCalibration = &gestureCalibration;
       }  else
       {
         fprintf(stderr,"Could not get Calibration\n");
         activeGestureCalibration =0;
         return 0;
       }

  return 1;
}



void closeEverything()
{
 fprintf(stderr,"Gracefully closing everything .. ");

 fubgtUpperBodyTracker_Close();

 //Stop our target ( can be network or files or nothing )
 acquisitionStopTargetForFrames(moduleID,devID);
 /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
 acquisitionCloseDevice(moduleID,devID);
 acquisitionStopModule(moduleID);

 fprintf(stderr,"Done\n");
 exit(0);
}

/**
 @brief this is the main function
 @ return nothing
 */
int main(int argc, char *argv[])
{
 fprintf(stderr,"Generic Grabber Application based on Acquisition lib .. \n");
 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);

// acquisitionRegisterTerminationSignal(&closeEverything);

  if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 1;
    }

  unsigned int width=640,height=480,framerate=30;
  unsigned int frameNum=0,maxFramesToGrab=0;
  int i=0;
  for (i=0; i<argc; i++)
  {

    if (strcmp(argv[i],"-v")==0)     { fubgtUpperBodyTracker_setVisualization(1); } else
    if (strcmp(argv[i],"-delay")==0) {
                                       delay=atoi(argv[i+1]);
                                       fprintf(stderr,"Delay set to %u seconds \n",delay);
                                      } else
    if (strcmp(argv[i],"-resolution")==0) {
                                             width=atoi(argv[i+1]);
                                             height=atoi(argv[i+2]);
                                             fubgtUpperBodyTracker_setSamplingFrameSize(width,height);
                                             fprintf(stderr,"Resolution set to %u x %u \n",width,height);
                                           } else
    if (strcmp(argv[i],"-calibration")==0) {
                                             calibrationSet=1;
                                             if (!ReadCalibration(argv[i+1],width,height,&calib) )
                                             {
                                               fprintf(stderr,"Could not read calibration file `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module")==0)    {
                                           moduleID = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleNameFromModuleID(moduleID),moduleID);
                                         } else
    if (strcmp(argv[i],"-dev")==0)      {
                                           devID = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID);
                                         } else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname); }
      else
    if (strcmp(argv[i],"-fps")==0)       {
                                            if (moduleID!=OPENNI2_ACQUISITION_MODULE)
                                            {
                                             framerate=atoi(argv[i+1]);
                                            } else
                                            {
                                              fprintf(stderr,"Not changing input device framerate because OpenNI2 will react badly :P \n");
                                            }
                                             float fpsFl = atof(argv[i+1]);
                                             fubgtUpperBodyTracker_setSamplingFrameRate(fpsFl,1 /*BLOCK to achieve sample rate*/);
                                             fprintf(stderr,"Framerate , set to %u  \n",framerate);
                                         }
  }


  if (!acquisitionIsModuleAvailiable(moduleID))
   {
       fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(moduleID));
       return 1;
   }

  //We want to check if deviceID we requested is a logical value , or we dont have that many devices!
  unsigned int maxDevID=acquisitionGetModuleDevices(moduleID);
  if ( (maxDevID==0) && (!acquisitionMayBeVirtualDevice(moduleID,devID,inputname)) ) { fprintf(stderr,"No devices availiable , and we didn't request a virtual device\n");  return 1; }
  if ( maxDevID < devID ) { fprintf(stderr,"Device Requested ( %u ) is out of range ( only %u available devices ) \n",devID,maxDevID);  return 1; }
  //If we are past here we are good to go..!



   if ( calibrationSet )
   {
    acquisitionSetColorCalibration(moduleID,devID,&calib);
    acquisitionSetDepthCalibration(moduleID,devID,&calib);
   }

   if (sizeof(struct calibration)!=sizeof(struct calibrationFUBGT))
    {
      fprintf(stderr,"Warning : Please note that RGBDAcquisition calibrations are no longer the same with internal representation\n");
    }



  char * devName = inputname;
  if (strlen(inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        if (!acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate))
        {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",devID,devName,getModuleNameFromModuleID(moduleID));
          return 1;
        }

        if ( strstr(inputname,"noRegistration")!=0 )         {  } else
        if ( strstr(inputname,"rgbToDepthRegistration")!=0 ) { acquisitionMapRGBToDepth(moduleID,devID); } else
                                                             { acquisitionMapDepthToRGB(moduleID,devID);  }

         //We initialize the target for our frames ( can be network or files )
         if (! acquisitionInitiateTargetForFrames(moduleID,devID,outputfoldername) )
         {
           return 1;
         }

   unsigned int colorWidth,colorHeight,colorChannels,colorBitsperpixel;
   acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);
   unsigned int depthWidth,depthHeight,depthChannels,depthBitsperpixel;
   acquisitionGetDepthFrameDimensions(moduleID,devID,&depthWidth,&depthHeight,&depthChannels,&depthBitsperpixel);


   fubgtUpperBodyTracker_Initialize(colorWidth,colorHeight,"../",argc,argv);
   fubgtUpperBodyTracker_useGestures(1);

   //fubgtUpperBodyTracker_RegisterSkeletonDetectedEvent((void *) &broadcastNewSkeleton);
    fubgtUpperBodyTracker_RegisterGestureDetectedEvent((void *) &broadcastNewGesture);

   countdownDelay(delay);
   fprintf(stderr,"Starting Grabbing!\n");



   while  ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) )
    {
        acquisitionSnapFrames(moduleID,devID);
        updateCalibration();

        fubgtUpperBodyTracker_NewFrame        (acquisitionGetColorFrame(moduleID,devID) , colorWidth,colorHeight ,
                                               acquisitionGetDepthFrame(moduleID,devID) , depthWidth,depthHeight ,
                                               activeGestureCalibration,
                                               0,
                                               frameNum);
        ++frameNum;
    }

    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);

    closeEverything();

    return 0;
}
