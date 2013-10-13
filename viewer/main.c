#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../tools/Calibration/calibration.h"

char inputname[512]={0};
unsigned int frameNum=0;


int calibrationSet = 0;
struct calibration calib;

//#include <opencv2/opencv.hpp>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>


int acquisitionDisplayFrames(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int framerate)
{


    //GIVE TIME FOR REDRAW EVENTS ETC -------------------------------------------------------------------------
    float msWaitTime = ((float) 1000/framerate) ;
    int key = cvWaitKey(msWaitTime );
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
                      exit (0);
                  break;
				}
			}





    unsigned int width , height , channels , bitsperpixel;

    //DRAW RGB FRAME -------------------------------------------------------------------------------------
    acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    IplImage  *imageRGB = cvCreateImage( cvSize(width , height), IPL_DEPTH_8U ,channels);
    if (imageRGB==0) { fprintf(stderr,"Could not create a new RGB OpenCV Image\n");  return 0; }
    char * opencv_color_pointer_retainer = imageRGB->imageData; // UGLY HACK
    imageRGB->imageData = (char *) acquisitionGetColorFrame(moduleID,devID);
    if ( (bitsperpixel==8) && (channels==3) ) { cvCvtColor( imageRGB, imageRGB, CV_RGB2BGR); }
    cvShowImage("RGBDAcquisition RGB ",imageRGB);
    imageRGB->imageData = opencv_color_pointer_retainer; // UGLY HACK
    cvReleaseImage( &imageRGB );


    //DRAW DEPTH FRAME -------------------------------------------------------------------------------------
    acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    IplImage  *imageDepth = cvCreateImage( cvSize(width , height), IPL_DEPTH_16U ,channels);
    if (imageDepth==0) { fprintf(stderr,"Could not create a new Depth OpenCV Image\n");  return 0; }
    char *opencv_depth_pointer_retainer = imageDepth->imageData; // UGLY HACK
    imageDepth->imageData = (char *) acquisitionGetDepthFrame(moduleID,devID);


    IplImage *rdepth8  = cvCreateImage(cvSize(width , height), IPL_DEPTH_8U, 1);
    cvConvertScaleAbs(imageDepth, rdepth8, 255.0/2048,0);
    cvShowImage("RGBDAcquisition Depth", rdepth8);
    cvReleaseImage( &rdepth8 );

    //cvShowImage("RGBDAcquisition Depth RAW",imageDepth);
    imageDepth->imageData = opencv_depth_pointer_retainer; // UGLY HACK
    cvReleaseImage( &imageDepth );



  return 1;
}


int main(int argc, char *argv[])
{
 fprintf(stderr,"Generic Grabber Application based on Acquisition lib .. \n");
 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);


  ModuleIdentifier moduleID = OPENGL_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

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
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleStringName(moduleID),moduleID);
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

  if (!acquisitionIsModuleLinked(moduleID))
   {
       fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module %s ..\n",getModuleStringName(moduleID));
       return 1;
   }

  //We want to initialize all possible devices in this example..
  unsigned int devID=0,maxDevID=acquisitionGetModuleDevices(moduleID);
  if (maxDevID==0)
  {
      fprintf(stderr,"No devices found for Module used \n");
      return 1;
  }


   if ( calibrationSet )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",calib.farPlane,calib.nearPlane);
    acquisitionSetColorCalibration(moduleID,devID,&calib);
    acquisitionSetDepthCalibration(moduleID,devID,&calib);
   }

  char * devName = inputname;
  if (strlen(inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate);

        if ( strstr(inputname,"noRegistration")!=0 )         {  } else
        if ( strstr(inputname,"rgbToDepthRegistration")!=0 ) { acquisitionMapRGBToDepth(moduleID,devID); } else
                                                             { acquisitionMapDepthToRGB(moduleID,devID);  }

        //acquisitionMapDepthToRGB(moduleID,devID);
        //acquisitionMapRGBToDepth(moduleID,devID);
     }

    char outfilename[512]={0};

   if ( maxFramesToGrab==0 ) { maxFramesToGrab= 1294967295; } //set maxFramesToGrab to "infinite" :P
   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {
     for (devID=0; devID<maxDevID; devID++)
      {
        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID,devID);

        acquisitionDisplayFrames(moduleID,devID,framerate);

        acquisitionStopTimer(0);
        if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));

      }
    }

    fprintf(stderr,"Done viewing %u frames! \n",maxFramesToGrab);

    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionCloseDevice(moduleID,devID);
     }

    acquisitionStopModule(moduleID);
    return 0;
}
