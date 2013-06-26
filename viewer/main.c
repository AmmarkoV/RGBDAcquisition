#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"

char inputname[512]={0};


#include <cv.h>
#include <cxcore.h>
#include <highgui.h>


int RGB2BGR(char * pixels , unsigned int width , unsigned int height)
{
  if ( (pixels==0) || (width==0) || (height==0)) { return 0; }
  char * curPixel = pixels;
  char * limitPixel = pixels + ( (width-1) * height * 3 );

  char * r ;
  char * g ;
  char tmp;
  char * b ;
  while (curPixel < limitPixel)
  {
    r = curPixel++;
    g = curPixel++;
    b = curPixel++;

    tmp = *r;
    *r = *b;
    *b = tmp;
  }

 return 1;
}



int acquisitionDisplayFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  unsigned int depth=IPL_DEPTH_8U;
  //if (img->bitsPerPixel==8) { depth=IPL_DEPTH_8U; } else
  //if (img->bitsPerPixel==16) { depth=IPL_DEPTH_16U; } else
  //                            { fprintf(stderr,"viewImage called with incorrect depth\n");  return 0; }

/*  IplImage  *image = cvCreateImage( cvSize(img->width,img->height), depth,img->channels);
  if (image==0) { fprintf(stderr,"Could not create a new OpenCV Image\n");  return 0; }

  char * opencv_color_pointer_retainer = image->imageData; // UGLY HACK
  image->imageData = (char *) img->pixels;

  if ( (img->bitsPerPixel==8) && (img->channels==3) ) { cvCvtColor( image, image, CV_RGB2BGR); }

  cvShowImage(name,image);
  cvWaitKey(4);

  image->imageData = opencv_color_pointer_retainer; // UGLY HACK
  cvReleaseImage( &image );*/
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

  unsigned int width=640,height=480,framerate=25;
  unsigned int frameNum=0,maxFramesToGrab=10;
  int i=0;
  for (i=0; i<argc; i++)
  {
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


  char * devName = inputname;
  if (strlen(inputname)<1) { devName=0; }
    //Initialize Every OpenNI Device
    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate);
        acquisitionMapDepthToRGB(moduleID,devID);
        //acquisitionMapRGBToDepth(moduleID,devID);
     }

    char outfilename[512]={0};

   if ( maxFramesToGrab==0 ) { maxFramesToGrab= 1294967295; } //set maxFramesToGrab to "infinite" :P
   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {

     for (devID=0; devID<maxDevID; devID++)
      {
        acquisitionSnapFrames(moduleID,devID);

        acquisitionDisplayFrames(moduleID,devID);
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
