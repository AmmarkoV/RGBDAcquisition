#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"


char outputfoldername[512]={0};


int makepath(char * path)
{
    // FILE *fp;
    /* Open the command for reading. */
    char command[1024];
    sprintf(command,"mkdir -p %s",outputfoldername);
    fprintf(stderr,"Executing .. %s \n",command);

    return system(command);
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

    //We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
    unsigned int frameNum=0,maxFramesToGrab=10;
    if (argc>1)
     {
          maxFramesToGrab=atoi(argv[1]);
          fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
     }
    if (argc>2)
     {
          moduleID = getModuleIdFromModuleName(argv[2]);
          fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleStringName(moduleID),moduleID);
     }

    strcpy(outputfoldername,"frames/");
    if (argc>3)
     {
          strcat(outputfoldername,argv[3]);
          makepath(outputfoldername);
          fprintf(stderr,"OutputPath , set to %s  \n",outputfoldername);
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

   if ( maxFramesToGrab==0 ) { maxFramesToGrab= 1294967295; } //set maxFramesToGrab to "infinite" :P
   for (frameNum=0; frameNum<maxFramesToGrab; frameNum++)
    {

    for (devID=0; devID<maxDevID; devID++)
      {
        acquisitionSnapFrames(moduleID,devID);

        sprintf(outfilename,"%s/colorFrame_%u_%05u.pnm",outputfoldername,devID,frameNum);
        acquisitionSaveColorFrame(moduleID,devID,outfilename);

        sprintf(outfilename,"%s/depthFrame_%u_%05u.pnm",outputfoldername,devID,frameNum);
        acquisitionSaveDepthFrame(moduleID,devID,outfilename);

        sprintf(outfilename,"%s/pointCloud_%u_%05u.pcd",outputfoldername,devID,frameNum);
        acquisitionSavePCDPointCoud(moduleID,devID,outfilename);

        //sprintf(outfilename,"%s/depthFrame1C_%u_%05u.pnm",outputfoldername,devID,frameNum);
        //acquisitionSaveDepthFrame1C(moduleID,devID,outfilename);

        //sprintf(outfilename,"%s/coloreddepthFrame_%u_%05u.pnm",outputfoldername,devID,frameNum);
        //acquisitionSaveColoredDepthFrame(moduleID,devID,outfilename);
      }
    }

    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);

    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionCloseDevice(moduleID,devID);
     }

    acquisitionStopModule(moduleID);

    return 0;
}
