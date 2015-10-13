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



volatile int stop=0;
unsigned char warnNoDepth=0,verbose=0;

unsigned int moveWindowX=0,moveWindowY=0,doMoveWindow=0;
unsigned int windowX=0,windowY=0;
unsigned int drawColor=1;
unsigned int drawDepth=1;

char inputname[512]={0};
unsigned int frameNum=0;
unsigned int seekFrame=0;
unsigned int loopFrame=0;

char RGBwindowName[250]={0};
char DepthwindowName[250]={0};

int calibrationSet = 0;
struct calibration calib;

#define UNINITIALIZED_DEVICE 66666
  unsigned int devID=0;
  unsigned int devID2=UNINITIALIZED_DEVICE;
  ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//



#if INTERCEPT_MOUSE_IN_WINDOWS
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
if  ( event == EVENT_LBUTTONDOWN )
    {
     fprintf(stderr,"Left button of the mouse is clicked - position (%u,%u)\n",x,y);
     float x3D,y3D,z3D;
     acquisitionGetDepth3DPointAtXYCameraSpace(moduleID,devID,x,y,&x3D,&y3D,&z3D);
     fprintf(stderr,"acquisitionGetDepthValueAtXY(%u,%u) = %u \n",x,y,acquisitionGetDepthValueAtXY(moduleID,devID,x,y));
     fprintf(stderr,"acquisitionGetDepth3DPointAtXYCameraSpace(%u,%u) = %0.2f , %0.2f , %0.2f\n",x,y,x3D,y3D,z3D);

    } else
if  ( event == EVENT_RBUTTONDOWN ) { fprintf(stderr,"Right button of the mouse is clicked - position (%u,%u)\n",x,y); } else
if  ( event == EVENT_MBUTTONDOWN ) { fprintf(stderr,"Middle button of the mouse is clicked - position (%u,%u)\n",x,y); }
// Commented out because it spams a lot -> else if  ( event == EVENT_MOUSEMOVE )   { fprintf(stderr,"Mouse move over the window - position (%u,%u)\n",x,y); }
}
#endif


int acquisitionStopDisplayingFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  	 cvDestroyWindow(RGBwindowName);
   	 cvDestroyWindow(DepthwindowName);
   	 return 1;
}




void closeEverything()
{
 fprintf(stderr,"Gracefully closing everything .. ");

 acquisitionStopDisplayingFrames(moduleID,devID);
 /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
 acquisitionCloseDevice(moduleID,devID);

 if (devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionCloseDevice(moduleID,devID2);
        }

 acquisitionStopModule(moduleID);

 fprintf(stderr,"Done\n");
 exit(0);
}



int acquisitionDisplayFrames(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int framerate)
{
    //GIVE TIME FOR REDRAW EVENTS ETC -------------------------------------------------------------------------
    float msWaitTime = ((float) 1000/framerate) ;
    int key = cvWaitKey(msWaitTime/3);
    if (key != -1)
			{
                char outfilename[1024]={0};
				key = 0x000000FF & key;
				switch (key)
				{
                  case 'S' :
				  case 's' :
				      fprintf(stderr,"S Key pressed , Saving Color/Depth frames\n");
                      sprintf(outfilename,"colorFrame_%u_%05u",devID,frameNum);
                      acquisitionSaveColorFrame(moduleID,devID,outfilename);

                      sprintf(outfilename,"depthFrame_%u_%05u",devID,frameNum);
                      acquisitionSaveDepthFrame(moduleID,devID,outfilename);
                      ++frameNum;
                  break;

                  case 'Q' :
				  case 'q' :
				      fprintf(stderr,"Q Key pressed , quitting\n");
                      stop=1; // exit (0);
                  break;
				}
			}



    unsigned int colorWidth = 640;
    unsigned int width , height , channels , bitsperpixel;


if (drawColor)
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

       if ( (windowX!=0) && (windowY!=0) )
          {
            //Enforce a different window size for the rgb window ( so we also need to rescale )
            CvSize rescaleSize = cvSize(windowX,windowY);
            IplImage* rescaledWindow=cvCreateImage(rescaleSize,IPL_DEPTH_8U ,channels);
            cvResize(imageViewableBGR,rescaledWindow,CV_INTER_LINEAR);
            cvShowImage(RGBwindowName,rescaledWindow);
            cvReleaseImage( &rescaledWindow );
          } else
          {
            cvShowImage(RGBwindowName,imageViewableBGR);
          }

        if (doMoveWindow)
          {  cvMoveWindow(RGBwindowName,moveWindowX,moveWindowY); }

      } else
      {
       fprintf(stderr,RED "Will not view RGB Frame , it is empty \n" NORMAL);
      }
     imageRGB->imageData = opencv_color_pointer_retainer; // UGLY HACK
     cvReleaseImage( &imageRGB );
     cvReleaseImage( &imageViewableBGR );
    }
}


if (drawDepth)
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
       if ( (windowX!=0) && (windowY!=0) )
            { //Enforce a different window size for the depth window ( so we also need to rescale )
              CvSize rescaleSize = cvSize(windowX,windowY);
              IplImage* rescaledWindow=cvCreateImage(rescaleSize,IPL_DEPTH_8U , 1);
              cvResize(rdepth8,rescaledWindow,CV_INTER_LINEAR);
              cvShowImage(DepthwindowName,rescaledWindow);
              cvReleaseImage( &rescaledWindow );
            } else
            {
              cvShowImage(DepthwindowName, rdepth8);
            }


        if (doMoveWindow)
          {
             unsigned int depthWindowX=windowX;
             if (depthWindowX==0) { depthWindowX=colorWidth; }
              cvMoveWindow(DepthwindowName,moveWindowX+depthWindowX,moveWindowY); }

       cvReleaseImage( &rdepth8 );
      } else
      {
       if (!warnNoDepth)
         { fprintf(stderr,RED "\n\n\n\nWill not view Depth Frame , it is empty [ This warning will only appear once ]\n\n\n\n" NORMAL); }
       warnNoDepth=1;
      }

     //cvShowImage("RGBDAcquisition Depth RAW",imageDepth);
     imageDepth->imageData = opencv_depth_pointer_retainer; // UGLY HACK
     cvReleaseImage( &imageDepth );
    }
}

  return 1;
}


int main(int argc, char *argv[])
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

  unsigned int width=640,height=480,framerate=30;
  unsigned int frameNum=0,maxFramesToGrab=0;
  int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-resolution")==0) {
                                             width=atoi(argv[i+1]);
                                             height=atoi(argv[i+2]);
                                             fprintf(stderr,"Resolution set to %u x %u \n",width,height);
                                           } else
    if (strcmp(argv[i],"-moveWindow")==0) {
                                           moveWindowX=atoi(argv[i+1]);
                                           moveWindowY=atoi(argv[i+2]);
                                           doMoveWindow=1;
                                           fprintf(stderr,"Setting window position to %u %u\n",moveWindowX,moveWindowY);
                                         } else
    if (strcmp(argv[i],"-resizeWindow")==0) {
                                             windowX=atoi(argv[i+1]);
                                             windowY=atoi(argv[i+2]);
                                             fprintf(stderr,"Window Sizes set to %u x %u \n",windowX,windowY);
                                           } else
    if (strcmp(argv[i],"-v")==0)           {
                                             verbose=1;
                                           } else
     if ( (strcmp(argv[i],"-onlyDepth")==0)||
          (strcmp(argv[i],"-noColor")==0)) {
                                               drawColor = 0;
                                           } else
     if ( (strcmp(argv[i],"-onlyColor")==0)||
          (strcmp(argv[i],"-noDepth")==0)) {
                                               drawDepth = 0;
                                           } else
    if (strcmp(argv[i],"-calibration")==0) {
                                             calibrationSet=1;
                                             if (!ReadCalibration(argv[i+1],width,height,&calib) )
                                             {
                                               fprintf(stderr,"Could not read calibration file `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-seek")==0)      {
                                           seekFrame=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting seek to %u \n",seekFrame);
                                         } else

    if (strcmp(argv[i],"-loop")==0)      {
                                           loopFrame=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting loop to %u \n",loopFrame);
                                         } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module")==0)    {
                                           moduleID = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleNameFromModuleID(moduleID),moduleID);
                                         } else
    if ( (strcmp(argv[i],"-dev")==0) ||
         (strcmp(argv[i],"-dev1")==0) )
                                         {
                                           devID = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID);
                                         } else
    if   (strcmp(argv[i],"-dev2")==0)    {
                                           devID2 = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device #2 Used , set to %s ( %u ) \n",argv[i+1],devID2);
                                         } else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname); }
      else
    if (strcmp(argv[i],"-fps")==0)       {
                                             framerate=atoi(argv[i+1]);
                                             fprintf(stderr,"Framerate , set to %u  \n",framerate);
                                         }
  }

  if (framerate==0) { fprintf(stderr,"Zero is an invalid value for framerate , using 1\n"); framerate=1; }



  if (drawColor==0) { acquisitionDisableStream(moduleID,devID,0); }
  if (drawDepth==0) { acquisitionDisableStream(moduleID,devID,1); }


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
    fprintf(stderr,"Set Far/Near to %f/%f\n",calib.farPlane,calib.nearPlane);
    acquisitionSetColorCalibration(moduleID,devID,&calib);
    acquisitionSetDepthCalibration(moduleID,devID,&calib);
   }

  char * devName = inputname;
  if (strlen(inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
      if (seekFrame!=0)
      {
          acquisitionSeekFrame(moduleID,devID,seekFrame);
      }
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        if (!acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate) )
        {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",devID,devName,getModuleNameFromModuleID(moduleID));
          return 1;
        }
        if ( strstr(inputname,"noRegistration")!=0 )         {  } else
        if ( strstr(inputname,"rgbToDepthRegistration")!=0 ) { acquisitionMapRGBToDepth(moduleID,devID); } else
                                                             { acquisitionMapDepthToRGB(moduleID,devID);  }
        fprintf(stderr,"Done with Mapping Depth/RGB \n");

        if (devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionOpenDevice(moduleID,devID2,devName,width,height,framerate);
        }


     sprintf(RGBwindowName,"RGBDAcquisition RGB - Module %u Device %u",moduleID,devID);
     sprintf(DepthwindowName,"RGBDAcquisition Depth - Module %u Device %u",moduleID,devID);

      if (seekFrame!=0)
      {
          acquisitionSeekFrame(moduleID,devID,seekFrame);
      }

     #if INTERCEPT_MOUSE_IN_WINDOWS
      //Create a window
       cvNamedWindow(RGBwindowName, 1);
      //set the callback function for any mouse event
       cvSetMouseCallback(RGBwindowName, CallBackFunc, NULL);
     #endif

   while ( (!stop) && ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) ) )
    {
        if (verbose)
        {
           fprintf(stderr,"Frame Number is : %u\n",frameNum);
        }
        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID,devID);

        acquisitionDisplayFrames(moduleID,devID,framerate);

       if (devID2!=UNINITIALIZED_DEVICE)
        {
          acquisitionSnapFrames(moduleID,devID2);
          acquisitionDisplayFrames(moduleID,devID2,framerate);
        }

        acquisitionStopTimer(0);
        if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
        ++frameNum;



        if (loopFrame!=0)
        {
          //fprintf(stderr,"%u%%(%u+%u)==%u\n",frameNum,loopFrame,seekFrame,frameNum%(loopFrame+seekFrame));
          if ( frameNum%(loopFrame)==0)
          {
            fprintf(stderr,"Looping Dataset , we reached frame %u ( %u ) , going back to %u\n",frameNum,loopFrame,seekFrame);
            acquisitionSeekFrame(moduleID,devID,seekFrame);
          }
        }

    }

    fprintf(stderr,"Done viewing %u frames! \n",frameNum);

    closeEverything();

    return 0;
}
