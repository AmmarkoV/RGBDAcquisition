#ifndef VIEWERSETTINGS_H_INCLUDED
#define VIEWERSETTINGS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

 unsigned int width,height,framerate;
 unsigned int maxFramesToGrab;
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
  vs->moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//


  vs->width=640; vs->height=480; vs->framerate=30;
  vs->frameNum=0; vs->maxFramesToGrab=0;
}



static int initializeViewerSettingsFromArguments(struct viewerSettings * vs,int argc,const char *argv[])
{
  fprintf(stderr,"Generic Grabber Application based on Acquisition lib .. \n");
  unsigned int possibleModules = acquisitionGetModulesCount();
  fprintf(stderr,"Linked to %u modules.. \n",possibleModules);

  if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 0;
    }


  initializeViewerSettings(vs);

  unsigned int i;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-delay")==0) { vs->delay=atoi(argv[i+1]); } else
    if (strcmp(argv[i],"-saveEveryFrame")==0) {  vs->saveEveryFrame=1; } else
    if (strcmp(argv[i],"-saveAsOriginalFrameNumber")==0) {
                                                           vs->saveAsOriginalFrameNumber=1;
                                                         } else
    if (strcmp(argv[i],"-nolocation")==0) {
                                            acquisitionSetLocation(vs->moduleID,0);
                                          } else
    if (strcmp(argv[i],"-executeEveryLoop")==0) {
                                                   fprintf(stderr,"Will Execute %s after each frame\n",argv[i+1]);
                                                   snprintf(vs->executeEveryLoop,1024,"%s",argv[i+1]);
                                                   vs->executeEveryLoopPayload=1;
                                                } else
    if (strcmp(argv[i],"-processor")==0) {
                                          fprintf(stderr,"Adding Processor to Pipeline %s , postfix %s\n",argv[i+1],argv[i+2]);
                                          if (!acquisitionAddProcessor(vs->moduleID,vs->devID,argv[i+1],argv[i+2],argc,argv))
                                          { fprintf(stderr,"Stopping execution..\n"); return 1; } else
                                          { fprintf(stderr,"Successfuly added processor..\n");  }
                                         } else
    if (strcmp(argv[i],"-waitKey")==0) {
                                         fprintf(stderr,"Waiting for key to be pressed to start\n");
                                         vs->waitKeyToStart=1;
                                        } else
    if (strcmp(argv[i],"-noinput")==0) {
                                         fprintf(stderr,"Disabling user input\n");
                                         vs->noinput=1;
                                        } else
    if (strcmp(argv[i],"-resolution")==0) {
                                             vs->width=atoi(argv[i+1]);
                                             vs->height=atoi(argv[i+2]);
                                             fprintf(stderr,"Resolution set to %u x %u \n",vs->width,vs->height);
                                           } else
    if (strcmp(argv[i],"-moveWindow")==0) {
                                           vs->moveWindowX=atoi(argv[i+1]);
                                           vs->moveWindowY=atoi(argv[i+2]);
                                           vs->doMoveWindow=1;
                                           fprintf(stderr,"Setting window position to %u %u\n",vs->moveWindowX,vs->moveWindowY);
                                         } else
    if (strcmp(argv[i],"-resizeWindow")==0) {
                                             vs->windowX=atoi(argv[i+1]);
                                             vs->windowY=atoi(argv[i+2]);
                                             fprintf(stderr,"Window Sizes set to %u x %u \n",vs->windowX,vs->windowY);
                                           } else
    if (strcmp(argv[i],"-v")==0)           {
                                             vs->verbose=1;
                                           } else
     if ( (strcmp(argv[i],"-onlyDepth")==0)||
          (strcmp(argv[i],"-noColor")==0)) {
                                               vs->drawColor = 0;
                                           } else
     if ( (strcmp(argv[i],"-onlyColor")==0)||
          (strcmp(argv[i],"-noDepth")==0)) {
                                               vs->drawDepth = 0;
                                           } else
    if (strcmp(argv[i],"-calibration")==0) {
                                             vs->calibrationSet=1;
                                             if (!ReadCalibration(argv[i+1],vs->width,vs->height,&vs->calib) )
                                             {
                                               fprintf(stderr,"Could not read calibration file `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-seek")==0)      {
                                           vs->seekFrame=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting seek to %u \n",vs->seekFrame);
                                         } else

    if (strcmp(argv[i],"-loop")==0)      {
                                           vs->loopFrame=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting loop to %u \n",vs->loopFrame);
                                         } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           vs->maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",vs->maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module")==0)    {
                                           vs->moduleID = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleNameFromModuleID(vs->moduleID),vs->moduleID);
                                         } else
    if ( (strcmp(argv[i],"-dev")==0) ||
         (strcmp(argv[i],"-dev1")==0) )
                                         {
                                           vs->devID = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],vs->devID);
                                         } else
    if   (strcmp(argv[i],"-dev2")==0)    {
                                           vs->devID2 = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device #2 Used , set to %s ( %u ) \n",argv[i+1],vs->devID2);
                                         } else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(vs->inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",vs->inputname); }
      else
    if (strcmp(argv[i],"-fps")==0)       {
                                             vs->framerate=atoi(argv[i+1]);
                                             fprintf(stderr,"Framerate , set to %u  \n",vs->framerate);
                                         }
  }




  if (vs->framerate==0) { fprintf(stderr,"Zero is an invalid value for framerate , using 1\n"); vs->framerate=1; }

  if (vs->moduleID==SCRIPTED_ACQUISITION_MODULE)
  {
  if (
      acquisitionGetScriptModuleAndDeviceID(
                                            &vs->moduleID ,
                                            &vs->devID ,
                                            &vs->width ,
                                            &vs->height,
                                            &vs->framerate,
                                            0,
                                            vs->inputname,
                                            512
                                           )
      )
      { fprintf(stderr,"Loaded configuration from script file..\n"); }
  }


  if (!acquisitionIsModuleAvailiable(vs->moduleID))
   {
       fprintf(stderr,"The module you are trying to use (%u) is not linked in this build of the Acquisition library..\n",vs->moduleID);
       return 1;
   }





 return 1;
}



static int acquisitionSaveFrames(
                          struct viewerSettings * config,
                          ModuleIdentifier moduleID,
                          DeviceIdentifier devID,
                          unsigned int framerate
                         )
{
 char outfilename[1024]={0};
    if (config->saveAsOriginalFrameNumber)
                      {
                        config->savedFrameNum=config->frameNum;
                        fprintf(stderr,"Saving %u using original Frame Numbers ( change by ommiting -saveAsOriginalFrameNumber ) \n",config->savedFrameNum);
                      } else
                      { fprintf(stderr,"Saving %u using seperate enumeration for saved images( change by using -saveAsOriginalFrameNumber ) \n",config->savedFrameNum); }

   if (config->drawColor)
                      {
                       sprintf(outfilename,"frames/colorFrame_%u_%05u",devID,config->savedFrameNum);
                       acquisitionSaveColorFrame(moduleID,devID,outfilename,config->compressOutput);
                      }

   if (config->drawDepth)
                      {
                       sprintf(outfilename,"frames/depthFrame_%u_%05u",devID,config->savedFrameNum);
                       acquisitionSaveDepthFrame(moduleID,devID,outfilename,config->compressOutput);
                      }
   ++config->savedFrameNum;
  return 1;
}



static void acquisitionDefaultTerminator(struct viewerSettings * config)
{
 fprintf(stderr,"Gracefully closing everything .. ");

 /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
 acquisitionCloseDevice(config->moduleID,config->devID);

 if (config->devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionCloseDevice(
                                  config->moduleID,
                                  config->devID2
                                );
        }

 acquisitionStopModule(config->moduleID);

 fprintf(stderr,"Done\n");
 exit(0);
}


#endif // VIEWERSETTINGS_H_INCLUDED
