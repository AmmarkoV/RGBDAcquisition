#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"


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

    //We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
    unsigned int frameNum=0,maxFramesToGrab=10;
    if (argc>1)
     {
          maxFramesToGrab=atoi(argv[1]);
          fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
     }
    if (argc>2)
     {
          if (strcmp("FREENECT",argv[2])==0 )  { moduleID = FREENECT_ACQUISITION_MODULE; } else
          if (strcmp("OPENNI1", argv[2])==0 )  { moduleID = OPENNI1_ACQUISITION_MODULE;  } else
          if (strcmp("OPENNI2", argv[2])==0 )  { moduleID = OPENNI2_ACQUISITION_MODULE;  } else
          if (strcmp("OPENGL", argv[2])==0 )   { moduleID = OPENGL_ACQUISITION_MODULE;   } else
          if (strcmp("TEMPLATE",argv[2])==0 )  { moduleID = TEMPLATE_ACQUISITION_MODULE; }

          fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleStringName(moduleID),moduleID);
     }

  if (!acquisitionIsModuleLinked(moduleID))
   {
       fprintf(stderr,"The module you are trying to use is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID,16 /*maxDevices*/))
  {
       fprintf(stderr,"Could not start module..\n");
       return 1;
   }

  //We want to initialize all possible devices in this example..
  unsigned int devID=0,maxDevID=acquisitionGetModuleDevices(moduleID);


  if (maxDevID==0)
  {
      fprintf(stderr,"No devices found for Module used \n");
      return 1;
  }

    //Initialize Every OpenNI Device
    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionOpenDevice(moduleID,devID,640,480,25);
        acquisitionMapDepthToRGB(moduleID,devID);
        //acquisitionMapRGBToDepth(moduleID,devID);
     }
    usleep(1000); // Waiting a while for the glitch frames to pass
    char outfilename[512]={0};

   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {

    for (devID=0; devID<maxDevID; devID++)
      {
        acquisitionSnapFrames(moduleID,devID);

        //fprintf(stderr,"Color frame is %ux%u:3 - %u \n",getOpenNI2ColorWidth(devID) , getOpenNI2ColorHeight(devID) , getOpenNI2ColorDataSize(devID));
        sprintf(outfilename,"frames/colorFrame_%u_%05u.pnm",devID,frameNum);
        acquisitionSaveColorFrame(moduleID,devID,outfilename);

        //fprintf(stderr,"Depth frame is %ux%u:1 - %u \n",getOpenNI2DepthWidth(devID) , getOpenNI2DepthHeight(devID) , getOpenNI2DepthDataSize(devID));
        sprintf(outfilename,"frames/depthFrame_%u_%05u.pnm",devID,frameNum);
        acquisitionSaveDepthFrame(moduleID,devID,outfilename);
      }
    }


    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionCloseDevice(moduleID,devID);
     }

    acquisitionStopModule(moduleID);

    return 0;
}
