#include "daemon_ammarserver.h"

#define USE_AMMARSERVER 1

#if USE_AMMARSERVER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../3dparty/AmmarServer/src/AmmServerlib/AmmServerlib.h"

#include "../tools/Codecs/codecs.h"
#include "../tools/Codecs/jpgInput.h"

#include "NetworkAcquisition.h"

#define MAX_JPEG_SIZE 128*1024*1024

char webserver_root[MAX_FILE_PATH]="src/Services/WebFramebuffer/"; // <- change this to the directory that contains your content if you dont want to use the default public_html dir..

char uploads_root[MAX_FILE_PATH]="uploads/";
char templates_root[MAX_FILE_PATH]="public_html/templates/";

unsigned int hits=0;

//The decleration of some dynamic content resources..
struct AmmServer_Instance  * default_server=0;

struct AmmServer_RH_Context indexContext={0};
struct AmmServer_RH_Context frameContext={0};


void * prepare_frame_content_callback(struct AmmServer_DynamicRequest  * rqst)
{
  ++hits;
  AmmServer_Success("in prepare_frame_content_callback");
  if ( _GETcmp(rqst,"stream","color") == 0 )
  {
     AmmServer_Success("Returning new color frame..");
     if (networkDevice[0].colorFrame!=0)
     {

      pthread_mutex_lock (&networkDevice[0].colorLock);   // LOCK PROTECTED OPERATION -------------------------------------------

      struct Image img = {0};
      populateImage(
                     &img,
                     networkDevice[0].colorWidth,
                     networkDevice[0].colorHeight,
                     networkDevice[0].colorChannels,
                     networkDevice[0].colorBitsperpixel,
                     networkDevice[0].colorFrame
                    );

      networkDevice[0].compressedColorSize = MAX_JPEG_SIZE;
      char * compressedPixels = (char* ) malloc(sizeof(char) * networkDevice[0].compressedColorSize);
      if ( WriteJPEGInternal("dummy.jpg",&img,compressedPixels,&networkDevice[0].compressedColorSize) )
      {
         AmmServer_Success("Successfully compressed JPEG frame..");
      } else
      {
         AmmServer_Warning("Could not compress JPEG frame..");
      }

      pthread_mutex_unlock (&networkDevice[0].colorLock);   // LOCK PROTECTED OPERATION -------------------------------------------

      memcpy(rqst->content,compressedPixels,networkDevice[0].compressedColorSize);
      rqst->contentSize=networkDevice[0].compressedColorSize;

      free(compressedPixels);

      AmmServer_Success("color frame ok..");
      return 0;
     } else
     {
         AmmServer_Warning("Color frame is empty..");
     }
  } else
  if ( _GETcmp(rqst,"stream","depth") == 0 )
  {
     AmmServer_Success("Returning new depth frame..");
     if (networkDevice[0].depthFrame!=0)
     {

      pthread_mutex_lock (&networkDevice[0].depthLock);   // LOCK PROTECTED OPERATION -------------------------------------------

       memcpy(rqst->content,networkDevice[0].depthFrame,networkDevice[0].depthFrameSize);
       rqst->contentSize=networkDevice[0].depthFrameSize;

      pthread_mutex_unlock (&networkDevice[0].depthLock);   // LOCK PROTECTED OPERATION -------------------------------------------

      AmmServer_Success("depth frame ok..");
      return 0;
     } else
     {
         AmmServer_Warning("Depth frame is empty..");
     }
  } else
  {
     unsigned int valueLength = 0;
     AmmServer_Warning("Incorrect frame stream requested (%s) ..",_GET(rqst,"stream",&valueLength));
  }


    snprintf(
             rqst->content,rqst->MAXcontentSize,"<!DOCTYPE html>\n<html>\
             <body>Could not find what to return..</body></html>"
             );
    rqst->contentSize=strlen(rqst->content);


  return 0;
}


void * prepare_index_content_callback(struct AmmServer_DynamicRequest  * rqst)
{
  snprintf(
    rqst->content,rqst->MAXcontentSize,"<!DOCTYPE html>\n<html>\
    <head>\
    <script>\
    function refreshFeed()\
    {\
      var randomnumber=Math.floor(Math.random()*100000);\
      document.getElementById(\"vfi\").style.visibility='visible';\
      document.getElementById(\"vfi\").src=\"framebuffer.jpg?stream=color&t=\"+randomnumber;\
    }\
     setInterval(refreshFeed,100);\
    </script>\
    </head>\
    <body><img  id=\"vfi\" src=\"framebuffer.jpg?stream=color&t=%u\"></body></html>",
    rand()%1000000
    );
  rqst->contentSize=strlen(rqst->content);
  return 0;
}



int ammarserver_UpdateFrameServerImages(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  if (streamNumber==0) //Color
  {
      networkDevice[0].okToSendColorFrame=0;
      networkDevice[0].colorWidth=width;
      networkDevice[0].colorHeight=height;
      networkDevice[0].colorChannels=channels;
      networkDevice[0].colorBitsperpixel=bitsperpixel;
      networkDevice[0].colorFrameSize=width*height*channels*(bitsperpixel/8); // Not using compression

      //networkDevice[0].colorFrame = (unsigned char*) pixels;

      pthread_mutex_lock (&networkDevice[0].colorLock);   // LOCK PROTECTED OPERATION -------------------------------------------

       if (networkDevice[0].colorFrame != 0 ) { free(networkDevice[0].colorFrame); networkDevice[0].colorFrame=0; }
       networkDevice[0].colorFrame = (unsigned char*) malloc(sizeof(char) * networkDevice[0].colorFrameSize);
       memcpy(networkDevice[0].colorFrame,pixels,networkDevice[0].colorFrameSize);

       networkDevice[0].compressedColorSize=0; // Not using compression
      pthread_mutex_unlock (&networkDevice[0].colorLock); // LOCK PROTECTED OPERATION -------------------------------------------

      return 1;
  }
   else
 if (streamNumber==1) //Depth
  {
      networkDevice[0].okToSendDepthFrame=0;
      networkDevice[0].depthWidth=width;
      networkDevice[0].depthHeight=height;
      networkDevice[0].depthChannels=channels;
      networkDevice[0].depthBitsperpixel=bitsperpixel;
      networkDevice[0].depthFrameSize = width*height*channels*(bitsperpixel/8);

      //networkDevice[0].depthFrame = (unsigned short*) pixels;

      pthread_mutex_lock (&networkDevice[0].depthLock);   // LOCK PROTECTED OPERATION -------------------------------------------

      if (networkDevice[0].depthFrame != 0 ) { free(networkDevice[0].depthFrame); networkDevice[0].depthFrame=0; }
      networkDevice[0].depthFrame = (unsigned short*) malloc(sizeof(short) * networkDevice[0].depthFrameSize);
      memcpy(networkDevice[0].depthFrame,pixels,networkDevice[0].depthFrameSize);

      pthread_mutex_unlock (&networkDevice[0].depthLock); // LOCK PROTECTED OPERATION -------------------------------------------

      return 1;
  }

  return 0;
}



//This function adds a Resource Handler for the pages stats.html and formtest.html and associates stats , form and their callback functions
void init_dynamic_content()
{
  AmmServer_AddResourceHandler(default_server,&indexContext ,"/index.html" ,4096,0,&prepare_index_content_callback,SAME_PAGE_FOR_ALL_CLIENTS);
  AmmServer_AddResourceHandler(default_server,&frameContext ,"/framebuffer.jpg",MAX_JPEG_SIZE,0,&prepare_frame_content_callback,DIFFERENT_PAGE_FOR_EACH_CLIENT);


    pthread_mutex_init(&networkDevice[0].colorLock,0);

    pthread_mutex_init(&networkDevice[0].depthLock,0);
}

//This function destroys all Resource Handlers and free's all allocated memory..!
void close_dynamic_content()
{
  AmmServer_RemoveResourceHandler(default_server,&indexContext,1);
  AmmServer_RemoveResourceHandler(default_server,&frameContext,1);
}




int ammarserver_StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{
    printf("\nAmmar Server %s starting up..\n",AmmServer_Version());
    //Check binary and header spec
    AmmServer_CheckIfHeaderBinaryAreTheSame(AMMAR_SERVER_HTTP_HEADER_SPEC);
    //Register termination signal for when we receive SIGKILL etc
    AmmServer_RegisterTerminationSignal(&close_dynamic_content);


    //Kick start AmmarServer , bind the ports , create the threads and get things going..!
    default_server = AmmServer_StartWithArgs(
                                             "rgbdacquisition_publisher",
                                              0,0 , //The internal server will use the arguments to change settings
                                              //If you don't want this look at the AmmServer_Start call
                                              bindAddr,
                                              bindPort,
                                              0, /*This means we don't want a specific configuration file*/
                                              webserver_root,
                                              templates_root
                                              );


    if (!default_server) { AmmServer_Error("Could not start server , shutting down everything.."); exit(1); }

    //Create dynamic content allocations and associate context to the correct files
    init_dynamic_content();
    //stats.html and formtest.html should be availiable from now on..!

    if (AmmServer_Running(default_server))  { return 1; }
    return 0;
}



int ammarserver_StopFrameServer(unsigned int devID)
{
    //Delete dynamic content allocations and remove stats.html and formtest.html from the server
    close_dynamic_content();

    //Stop the server and clean state
    AmmServer_Stop(default_server);
    AmmServer_Warning("Ammar Server stopped\n");
 return 1;
}

#endif // USE_AMMARSERVER
