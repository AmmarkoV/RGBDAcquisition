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

#define INTERCEPT_MOUSE_IN_WINDOWS 1

#define NORMAL "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */


#define USE_NEW_OPENCV_HEADERS 1

#if USE_NEW_OPENCV_HEADERS
 //#include <opencv2/opencv.hpp>
 #include <opencv2/imgproc/imgproc_c.h>
 #include <opencv2/legacy/legacy.hpp>
 #include "opencv2/highgui/highgui.hpp"
#else
 #include <cv.h>
 #include <cxcore.h>
 #include <highgui.h>
#endif



#if INTERCEPT_MOUSE_IN_WINDOWS
enum
{
    EVENT_MOUSEMOVE      =0,
    EVENT_LBUTTONDOWN    =1,
    EVENT_RBUTTONDOWN    =2,
    EVENT_MBUTTONDOWN    =3,
    EVENT_LBUTTONUP      =4,
    EVENT_RBUTTONUP      =5,
    EVENT_MBUTTONUP      =6,
    EVENT_LBUTTONDBLCLK  =7,
    EVENT_RBUTTONDBLCLK  =8,
    EVENT_MBUTTONDBLCLK  =9
};

enum
{
    EVENT_FLAG_LBUTTON   =1,
    EVENT_FLAG_RBUTTON   =2,
    EVENT_FLAG_MBUTTON   =4,
    EVENT_FLAG_CTRLKEY   =8,
    EVENT_FLAG_SHIFTKEY  =16,
    EVENT_FLAG_ALTKEY    =32
};
#endif


struct viewerSettings config={0};



#if INTERCEPT_MOUSE_IN_WINDOWS
void CallBackFunc(struct viewerSettings * config,int event, int x, int y, int flags, void* userdata)
{
if  ( event == EVENT_LBUTTONDOWN )
    {
     fprintf(stderr,"Left button of the mouse is clicked - position (%u,%u)\n",x,y);
     float x3D,y3D,z3D;
     acquisitionGetDepth3DPointAtXYCameraSpace(config->moduleID,config->devID,x,y,&x3D,&y3D,&z3D);
     fprintf(stderr,"acquisitionGetDepthValueAtXY(%u,%u) = %u \n",x,y,acquisitionGetDepthValueAtXY(config->moduleID,config->devID,x,y));
     fprintf(stderr,"acquisitionGetDepth3DPointAtXYCameraSpace(%u,%u) = %0.2f , %0.2f , %0.2f\n",x,y,x3D,y3D,z3D);

    } else
if  ( event == EVENT_RBUTTONDOWN ) { fprintf(stderr,"Right button of the mouse is clicked - position (%u,%u)\n",x,y); } else
if  ( event == EVENT_MBUTTONDOWN ) { fprintf(stderr,"Middle button of the mouse is clicked - position (%u,%u)\n",x,y); }
// Commented out because it spams a lot -> else if  ( event == EVENT_MOUSEMOVE )   { fprintf(stderr,"Mouse move over the window - position (%u,%u)\n",x,y); }
}
#endif

int blockWaitingForKey()
{
    cvWaitKey(0); //block for ever until key pressed
 return 1;
}


int acquisitionStopDisplayingFrames(struct viewerSettings * config,ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  	 cvDestroyWindow(config->RGBwindowName);
   	 cvDestroyWindow(config->DepthwindowName);
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




int acquisitionDisplayFrames(
                              struct viewerSettings * config,
                              ModuleIdentifier moduleID,
                              DeviceIdentifier devID,
                              unsigned int framerate
                            )
{
    //GIVE TIME FOR REDRAW EVENTS ETC -------------------------------------------------------------------------
    if (config->saveEveryFrame)
     {
       acquisitionSaveFrames(config,moduleID,devID,framerate);
     }

    if (!config->noinput)
    {
    float msWaitTime = ((float) 1000/framerate) ;
    int key = cvWaitKey(msWaitTime/3);
    if (key != -1)
			{
				key = 0x000000FF & key;
				switch (key)
				{
                  case 'S' :
				  case 's' :
				      fprintf(stderr,"S Key pressed , Saving Color/Depth frames\n");
                      acquisitionSaveFrames(config,moduleID,devID,framerate);
                  break;

                  case 'Q' :
				  case 'q' :
				      fprintf(stderr,"Q Key pressed , quitting\n");
                      config->stop=1; // exit (0);
                  break;

                  case 32 :
                      if (config->waitKeyToStart==1) { config->waitKeyToStart=0; } else { config->waitKeyToStart=1; }
                  break;

                  default :
                      fprintf(stderr,"Unhandled key press %u \n",key);

                  break;
				}
			}

    } else
    {
     cvWaitKey(1);
    }


    unsigned int colorWidth = 640;
    unsigned int width , height , channels , bitsperpixel;


if (config->drawColor)
{
    //DRAW RGB FRAME -------------------------------------------------------------------------------------
    if ( acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel) )
    {
     colorWidth = width;
     IplImage  *imageRGB = cvCreateImage( cvSize(width , height), IPL_DEPTH_8U ,channels);
     IplImage  *imageViewableBGR = cvCreateImage( cvSize(width , height), IPL_DEPTH_8U ,channels);
     if (imageRGB==0) { fprintf(stderr,"Could not create a new RGB OpenCV Image\n");  return 0; }
     char * opencv_color_pointer_retainer = imageRGB->imageData; // UGLY HACK
     imageRGB->imageData = (char *) acquisitionGetColorFrame(moduleID,devID);
     if (imageRGB->imageData != 0)
      {

       //We convert RGB -> BGR @ imageViewableBFR so that we wont access original memory ,and OpenCV can happily display the correct colors etc
       if ( (bitsperpixel==8) && (channels==3) ) { cvCvtColor(imageRGB , imageViewableBGR , CV_RGB2BGR); }

       if ( (config->windowX!=0) && (config->windowY!=0) )
          {
            //Enforce a different window size for the rgb window ( so we also need to rescale )
            CvSize rescaleSize = cvSize(config->windowX,config->windowY);
            IplImage* rescaledWindow=cvCreateImage(rescaleSize,IPL_DEPTH_8U ,channels);
            cvResize(imageViewableBGR,rescaledWindow,CV_INTER_LINEAR);
            cvShowImage(config->RGBwindowName,rescaledWindow);
            cvReleaseImage( &rescaledWindow );
          } else
          {
            cvShowImage(config->RGBwindowName,imageViewableBGR);
          }

        if (config->doMoveWindow)
          {
             cvMoveWindow(
                           config->RGBwindowName,
                           config->moveWindowX,
                           config->moveWindowY
                         );
          }

      } else
      {
       fprintf(stderr,RED "Will not view RGB Frame , it is empty \n" NORMAL);
      }
     imageRGB->imageData = opencv_color_pointer_retainer; // UGLY HACK
     cvReleaseImage( &imageRGB );
     cvReleaseImage( &imageViewableBGR );
    }
}


if (config->drawDepth)
{
    //DRAW DEPTH FRAME -------------------------------------------------------------------------------------
    if ( acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel) )
    {
     IplImage  *imageDepth = cvCreateImage( cvSize(width , height), IPL_DEPTH_16U ,channels);
     if (imageDepth==0) { fprintf(stderr,"Could not create a new Depth OpenCV Image\n");  return 0; }
     char *opencv_depth_pointer_retainer = imageDepth->imageData; // UGLY HACK
     imageDepth->imageData = (char *) acquisitionGetDepthFrame(moduleID,devID);

     if (imageDepth->imageData != 0)
      {
       IplImage *rdepth8  = cvCreateImage(cvSize(width , height), IPL_DEPTH_8U, 1);
       cvConvertScaleAbs(imageDepth, rdepth8, 255.0/2048,0);
       if ( (config->windowX!=0) && (config->windowY!=0) )
            { //Enforce a different window size for the depth window ( so we also need to rescale )
              CvSize rescaleSize = cvSize(config->windowX,config->windowY);
              IplImage* rescaledWindow=cvCreateImage(rescaleSize,IPL_DEPTH_8U , 1);
              cvResize(rdepth8,rescaledWindow,CV_INTER_LINEAR);
              cvShowImage(config->DepthwindowName,rescaledWindow);
              cvReleaseImage( &rescaledWindow );
            } else
            {
              cvShowImage(config->DepthwindowName, rdepth8);
            }


        if (config->doMoveWindow)
          {
             unsigned int depthWindowX=config->windowX;
             if (depthWindowX==0) { depthWindowX=colorWidth; }
              cvMoveWindow(config->DepthwindowName,config->moveWindowX+depthWindowX,config->moveWindowY); }

       cvReleaseImage( &rdepth8 );
      } else
      {
       if (!config->warnNoDepth)
         { fprintf(stderr,RED "\n\n\n\nWill not view Depth Frame , it is empty [ This warning will only appear once ]\n\n\n\n" NORMAL); }
       config->warnNoDepth=1;
      }

     //cvShowImage("RGBDAcquisition Depth RAW",imageDepth);
     imageDepth->imageData = opencv_depth_pointer_retainer; // UGLY HACK
     cvReleaseImage( &imageDepth );
    }
}

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
         blockWaitingForKey();
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
