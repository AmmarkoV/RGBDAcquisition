#ifndef VIEWERSETTINGS_H_INCLUDED
#define VIEWERSETTINGS_H_INCLUDED


#define NORMAL "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */


#define UNINITIALIZED_DEVICE 66666
//#include "../tools/Calibration/calibration.h"

struct viewerSettings
{
 unsigned int stop;
 unsigned char warnNoDepth,verbose;

 unsigned int moveWindowX,moveWindowY,doMoveWindow;
 unsigned int windowX,windowY;
 unsigned int drawColor;
 unsigned int drawDepth;
 unsigned int noinput;
 unsigned int waitKeyToStart;

 char executeEveryLoop[1024];
 unsigned int executeEveryLoopPayload;

 char inputname[512];
 int compressOutput;
 int saveEveryFrame;
 int saveAsOriginalFrameNumber;
 unsigned int savedFrameNum;
 unsigned int frameNum;
 unsigned int seekFrame;
 unsigned int loopFrame;
 unsigned int delay;

 char RGBwindowName[250];
 char DepthwindowName[250];

 int calibrationSet;
 struct calibration calib;

 unsigned int devID;
 unsigned int devID2;
 ModuleIdentifier moduleID;
};




static void initializeViewerSettings(struct viewerSettings * vs)
{
  vs->stop=0;
  vs->warnNoDepth=0, vs->verbose=0;

  vs->moveWindowX=0, vs->moveWindowY=0, vs->doMoveWindow=0;
  vs->windowX=0, vs->windowY=0;
  vs->drawColor=1;
  vs->drawDepth=1;
  vs->noinput=0;
  vs->waitKeyToStart=0;

  vs->executeEveryLoop[0]=0;
  vs->executeEveryLoopPayload=0;

  vs->inputname[0]=0;
  vs->compressOutput=0;
  vs->saveEveryFrame=0;
  vs->saveAsOriginalFrameNumber=0;
  vs->savedFrameNum=0;
  vs->frameNum=0;
  vs->seekFrame=0;
  vs->loopFrame=0;
  vs->delay=0;

  vs->RGBwindowName[0]=0;
  vs->DepthwindowName[0]=0;

  vs->calibrationSet = 0;
  struct calibration calib={0};
  vs->calib=calib;

  #define UNINITIALIZED_DEVICE 66666
  vs->devID=0;
  vs->devID2=UNINITIALIZED_DEVICE;
  vs->moduleID = 0;//OPENNI1_ACQUISITION_MODULE;//

}






#endif // VIEWERSETTINGS_H_INCLUDED
