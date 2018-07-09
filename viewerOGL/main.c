/** @file main.c
 *  @brief The simple Viewer that uses libAcquisition.so to view input from a module/device
 *  This should be used like ./Viewer -module TEMPLATE -from Dataset
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug No known bugs
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../tools/Calibration/calibration.h"

#include "../tools/Common/viewerSettings.h"

// Include GLEW
#include <GL/glew.h>

//GLU
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <GL/glut.h>


struct viewerSettings config={0};


int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
    return 0;
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
    return 0;
}

#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/System/glx3.h"



int acquisitionCreateDisplay(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);

  if (!start_glx3_stuff(colorWidth,colorHeight,1,0,0)) { fprintf(stderr,"Could not initialize"); return 1;}


  if (glewInit() != GLEW_OK)
   {
		fprintf(stderr, "Failed to initialize GLEW\n");
	 	return 0;
   }
  return 1;
}

int acquisitionDisplayFrames(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int framerate)
{


 return 1;
}


int acquisitionStopDisplayingFrames(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   	 return 1;
}



void closeEverything(struct viewerSettings * config)
{
 fprintf(stderr,"Gracefully closing everything .. ");

 acquisitionStopDisplayingFrames(config,config->moduleID,config->devID);
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


int acquisitionSaveFrames(
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





int main(int argc,const char *argv[])
{
 fprintf(stderr,"Generic Grabber Application based on Acquisition lib .. \n");
 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);

 acquisitionRegisterTerminationSignal(&closeEverything);

  if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 1;
    }

  initializeViewerSettingsFromArguments(&config,argc,argv);

  if (config.framerate==0) { fprintf(stderr,"Zero is an invalid value for framerate , using 1\n"); config.framerate=1; }

  if (config.moduleID==SCRIPTED_ACQUISITION_MODULE)
  {
  if (
      acquisitionGetScriptModuleAndDeviceID(
                                            &config.moduleID ,
                                            &config.devID ,
                                            &config.width ,
                                            &config.height,
                                            &config.framerate,
                                            0,
                                            config.inputname,
                                            512
                                           )
      )
      { fprintf(stderr,"Loaded configuration from script file..\n"); }
  }


  if (!acquisitionIsModuleAvailiable(config.moduleID))
   {
       fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(config.moduleID,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module %s ..\n",getModuleNameFromModuleID(config.moduleID));
       return 1;
   }

  if (config.drawColor==0) { acquisitionDisableStream(config.moduleID,config.devID,0); }
  if (config.drawDepth==0) { acquisitionDisableStream(config.moduleID,config.devID,1); }


  //We want to check if deviceID we requested is a logical value , or we dont have that many devices!
  unsigned int maxDevID=acquisitionGetModuleDevices(config.moduleID);
  if ( (maxDevID==0) && (!acquisitionMayBeVirtualDevice(config.moduleID,config.devID,config.inputname)) ) { fprintf(stderr,"No devices availiable , and we didn't request a virtual device\n");  return 1; }
  if ( maxDevID < config.devID ) { fprintf(stderr,"Device Requested ( %u ) is out of range ( only %u available devices ) \n",config.devID,maxDevID);  return 1; }
  //If we are past here we are good to go..!


   if ( config.calibrationSet )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",config.calib.farPlane,config.calib.nearPlane);
    acquisitionSetColorCalibration(config.moduleID,config.devID,&config.calib);
    acquisitionSetDepthCalibration(config.moduleID,config.devID,&config.calib);
   }

  char * devName = config.inputname;
  if (strlen(config.inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
      if (config.seekFrame!=0)
      {
          acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
      }
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        if (!acquisitionOpenDevice(config.moduleID,config.devID,devName,config.width,config.height,config.framerate) )
        {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",config.devID,devName,getModuleNameFromModuleID(config.moduleID));
          return 1;
        }
        if ( strstr(config.inputname,"noRegistration")!=0 )         {  } else
        if ( strstr(config.inputname,"rgbToDepthRegistration")!=0 ) { acquisitionMapRGBToDepth(config.moduleID,config.devID); } else
                                                                    { acquisitionMapDepthToRGB(config.moduleID,config.devID); }
        fprintf(stderr,"Done with Mapping Depth/RGB \n");

        if (config.devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionOpenDevice(config.moduleID,config.devID2,devName,config.width,config.height,config.framerate);
        }


     sprintf(config.RGBwindowName,"RGBDAcquisition RGB - Module %u Device %u",config.moduleID,config.devID);
     sprintf(config.DepthwindowName,"RGBDAcquisition Depth - Module %u Device %u",config.moduleID,config.devID);

      if (config.seekFrame!=0)
      {
          acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
      }

     #if INTERCEPT_MOUSE_IN_WINDOWS
      //Create a window
    if (config.drawColor)
    {
       cvNamedWindow(config.RGBwindowName, 1);
      //set the callback function for any mouse event
       if (!config.noinput)
        {
         cvSetMouseCallback(config.RGBwindowName, CallBackFunc, NULL);
        }
     }
     #endif

   while ( (!config.stop) && ( (config.maxFramesToGrab==0)||(config.frameNum<config.maxFramesToGrab) ) )
    {
        if (config.verbose)
        {
           fprintf(stderr,"Frame Number is : %u\n",config.frameNum);
        }
        acquisitionStartTimer(0);

        acquisitionSnapFrames(config.moduleID,config.devID);

        acquisitionDisplayFrames(&config,config.moduleID,config.devID,config.framerate);

       if (config.devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionSnapFrames(config.moduleID,config.devID2);
          acquisitionDisplayFrames(&config,config.moduleID,config.devID2,config.framerate);
        }

        acquisitionStopTimer(0);
        if (config.frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
        ++config.frameNum;

       if ( config.waitKeyToStart>0 )
       {
         --config.waitKeyToStart;
       }

        if (config.loopFrame!=0)
        {
          //fprintf(stderr,"%u%%(%u+%u)==%u\n",frameNum,loopFrame,seekFrame,frameNum%(loopFrame+seekFrame));
          if ( config.frameNum%(config.loopFrame)==0)
          {
            fprintf(stderr,"Looping Dataset , we reached frame %u ( %u ) , going back to %u\n",config.frameNum,config.loopFrame,config.seekFrame);
            acquisitionSeekFrame(config.moduleID,config.devID,config.seekFrame);
          }
        }


      if (config.executeEveryLoopPayload)
      {
         int i=system(config.executeEveryLoop);
         if (i!=0) { fprintf(stderr,"Could not execute payload\n"); }
      }

      if (config.delay>0)
      {
         usleep(config.delay);
      }

    }

    fprintf(stderr,"Done viewing %u frames! \n",config.frameNum);

    closeEverything(&config);

    return 0;
}

