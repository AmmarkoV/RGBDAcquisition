#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "AmmarServer/src/AmmServerNULLlib/AmmServerlib.h"

#define DEFAULT_BINDING_IP "0.0.0.0"
#define DEFAULT_BINDING_PORT 8080 // <--- Change this to 80 if you want to bind to the default http port..!

#define MAX_RGB_FRAME_WIDTH 640
#define MAX_RGB_FRAME_HEIGHT 480

#define MAX_DEPTH_FRAME_WIDTH 640
#define MAX_DEPTH_FRAME_HEIGHT 480

ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE; //OPENGL_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

char webserver_root[MAX_FILE_PATH]="public_html/"; // <- change this to the directory that contains your content if you dont want to use the default public_html dir..
char templates_root[MAX_FILE_PATH]="public_html/templates/";

char outputfoldername[512]={0};

struct AmmServer_Instance * default_server=0;
struct AmmServer_RH_Context rgbFrame={0};
struct AmmServer_RH_Context depthFrame={0};


void * prepare_RGB_frame_content_callback(char * content)
{
  unsigned int width , height , channels , bitsperpixel , copySize;
  acquisitionGetColorFrameDimensions(moduleID,0,&width,&height,&channels,&bitsperpixel);
  copySize = width * height * channels * (bitsperpixel/8);
  strncpy(content,acquisitionGetColorFrame(moduleID,0),copySize);
  rgbFrame.content_size=copySize;
  return 0;
}

void * prepare_Depth_frame_content_callback(char * content)
{
  unsigned int width , height , channels , bitsperpixel , copySize;
  acquisitionGetDepthFrameDimensions(moduleID,0,&width,&height,&channels,&bitsperpixel);
  copySize = width * height * channels * (bitsperpixel/8);
  strncpy(content,acquisitionGetDepthFrame(moduleID,0),copySize);
  depthFrame.content_size=copySize;
  return 0;
}

void init_dynamic_content()
{
  unsigned int RGB_FRAME_SIZE =  MAX_RGB_FRAME_WIDTH * MAX_RGB_FRAME_HEIGHT * 3 ;
  if (! AmmServer_AddResourceHandler(default_server,&rgbFrame,"/rgb.raw",webserver_root,RGB_FRAME_SIZE,0,&prepare_RGB_frame_content_callback,SAME_PAGE_FOR_ALL_CLIENTS) ) { AmmServer_Warning("Failed adding rgbFrame page\n"); }

  unsigned int DEPTH_FRAME_SIZE =  MAX_DEPTH_FRAME_WIDTH * MAX_DEPTH_FRAME_HEIGHT * 2 ;
  if (! AmmServer_AddResourceHandler(default_server,&depthFrame,"/depth.raw",webserver_root,RGB_FRAME_SIZE,0,&prepare_Depth_frame_content_callback,SAME_PAGE_FOR_ALL_CLIENTS) ) { AmmServer_Warning("Failed adding depthFrame page\n"); }
}

//This function destroys all Resource Handlers and free's all allocated memory..!
void close_dynamic_content()
{
    AmmServer_RemoveResourceHandler(default_server,&rgbFrame,1);
    AmmServer_RemoveResourceHandler(default_server,&depthFrame,1);
}

int main(int argc, char *argv[])
{
 AmmServer_RegisterTerminationSignal(&close_dynamic_content);


 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);



 if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 1;
    }

    //We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
    unsigned int frameNum=0,maxFramesToGrab=10;


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

   default_server = AmmServer_Start ( DEFAULT_BINDING_IP, DEFAULT_BINDING_PORT, 0 /*don't want a configuration file*/ , webserver_root, templates_root );
   if (!default_server) { AmmServer_Error("Could not start server , shutting down everything.."); exit(1); }
   init_dynamic_content();
   while ( (AmmServer_Running(default_server)) )
   {
      //Do sampling here
    for (devID=0; devID<maxDevID; devID++)
      {
        acquisitionSnapFrames(moduleID,devID);
      }
     usleep(10000);
   }


    for (devID=0; devID<maxDevID; devID++)
     {
        /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
        acquisitionCloseDevice(moduleID,devID);
     }

    close_dynamic_content();
    acquisitionStopModule(moduleID);
    AmmServer_Stop(default_server);
    return 0;
}
