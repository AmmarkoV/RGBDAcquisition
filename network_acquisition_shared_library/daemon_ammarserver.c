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

char webserver_root[MAX_FILE_PATH]="src/Services/WebFramebuffer/"; // <- change this to the directory that contains your content if you dont want to use the default public_html dir..

char uploads_root[MAX_FILE_PATH]="uploads/";
char templates_root[MAX_FILE_PATH]="public_html/templates/";

unsigned int hits=0;


struct imageStorage
{
  char * data;
  unsigned int dataSize;
  unsigned int width;
  unsigned int height;
  unsigned int framenumber;
  unsigned int depth;

};


struct imageStorage storage[12]={0};

//The decleration of some dynamic content resources..
struct AmmServer_Instance  * default_server=0;

struct AmmServer_RH_Context indexContext={0};
struct AmmServer_RH_Context frameContext={0};

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
      document.getElementById(\"vfi\").src=\"framebuffer.jpg?t=\"+randomnumber;\
    }\
     setInterval(refreshFeed,100);\
    </script>\
    </head>\
    <body><img  id=\"vfi\" src=\"framebuffer.jpg?t=%u\"></body></html>",
    rand()%1000000
    );
  rqst->contentSize=strlen(rqst->content);
  return 0;
}





void * prepare_frame_content_callback(struct AmmServer_DynamicRequest  * rqst)
{
  ++hits;

  AmmServer_Success("Returning new frame..");
  memcpy(rqst->content,storage[0].data,storage[0].dataSize);
  rqst->contentSize=storage[0].dataSize;

  return 0;
}

//This function adds a Resource Handler for the pages stats.html and formtest.html and associates stats , form and their callback functions
void init_dynamic_content()
{
  AmmServer_AddResourceHandler(default_server,&indexContext ,"/index.html" ,4096,0,&prepare_index_content_callback,SAME_PAGE_FOR_ALL_CLIENTS);
  AmmServer_AddResourceHandler(default_server,&frameContext ,"/framebuffer.jpg",1128000,0,&prepare_frame_content_callback,DIFFERENT_PAGE_FOR_EACH_CLIENT);
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

         while ( (AmmServer_Running(default_server))  )
           {
             //Main thread should just sleep and let the background threads do the hard work..!
             //In other applications the programmer could use the main thread to do anything he likes..
             //The only caveat is that he would takeup more CPU time from the server and that he would have to poll
             //the AmmServer_Running() call once in a while to make sure everything is in order
             //usleep(60000);
             sleep(1);
           }

    //Delete dynamic content allocations and remove stats.html and formtest.html from the server
    close_dynamic_content();

    //Stop the server and clean state
    AmmServer_Stop(default_server);
    AmmServer_Warning("Ammar Server stopped\n");
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
      networkDevice[0].colorFrame = (unsigned char*) pixels;
      networkDevice[0].compressedColorSize=0; // Not using compression

      /*
      struct Image * img = createImageUsingExistingBuffer(width,height,channels,bitsperpixel,pixels);
      networkDevice[0].compressedColorSize=64*1024; //64KBmax
      char * compressedPixels = (char* ) malloc(sizeof(char) * networkDevice[0].compressedColorSize);
      WriteJPEGInternal("dummyName.jpg",img,compressedPixels,&networkDevice[0].compressedColorSize);
      networkDevice[0].colorFrame = (char*) compressedPixels;
      fprintf(stderr,"Compressed from %u bytes\n",networkDevice[0].compressedColorSize);*/


      networkDevice[0].okToSendColorFrame=1;


      while (networkDevice[0].okToSendColorFrame==1)
       {
         usleep(1000);
         fprintf(stderr,"Cf.");
       }
  }
   else
 if (streamNumber==1) //Depth
  {
      networkDevice[0].okToSendDepthFrame=0;
      networkDevice[0].depthWidth=width;
      networkDevice[0].depthHeight=height;
      networkDevice[0].depthChannels=channels;
      networkDevice[0].depthBitsperpixel=bitsperpixel;
      networkDevice[0].depthFrame = (unsigned short*) pixels;

      networkDevice[0].okToSendDepthFrame=1;


      while (networkDevice[0].okToSendDepthFrame==1)
       {
         usleep(1000);
         fprintf(stderr,"Df.");
       }
  }

  return 0;
}

void * ammarserver_ServeClient(void * ptr)
{
    /*
  fprintf(stderr,"Serve Client called ..\n");
  struct PassToHTTPThread * context = (struct PassToHTTPThread *) ptr;
  if (context->keep_var_on_stack!=1)
   {
     error("KeepVarOnStack is not properly set , this is a bug .. \n Will not serve request");
     fprintf(stderr,"Bad new thread context is pointing to %p\n",context);
     return 0;
   }
  int instanceID = context->id;
  int clientsock = context->clientsock;
  context->keep_var_on_stack=2;

  while (serverDevices[0].serverRunning)
   {
         if (networkDevice[0].okToSendColorFrame)
         {
           struct Image * img = createImageUsingExistingBuffer(networkDevice[0].colorWidth,networkDevice[0].colorHeight,networkDevice[0].colorChannels,
                                                               networkDevice[0].colorBitsperpixel,networkDevice[0].colorFrame);
           networkDevice[0].compressedColorSize=64*1024; //64KBmax
           char * compressedPixels = (char* ) malloc(sizeof(char) * networkDevice[0].compressedColorSize);
           WriteJPEGInternal("dummyName.jpg",img,compressedPixels,&networkDevice[0].compressedColorSize);
           networkDevice[0].colorFrame = (char*) compressedPixels;
           fprintf(stderr,"Compressed from %u bytes\n",networkDevice[0].compressedColorSize);


           sendImageSocket( clientsock ,networkDevice[0].colorFrame, networkDevice[0].colorWidth , networkDevice[0].colorHeight , networkDevice[0].colorChannels , networkDevice[0].colorBitsperpixel , networkDevice[0].compressedColorSize );

           free(compressedPixels);
           networkDevice[0].okToSendColorFrame=0;
         }

         if (networkDevice[0].okToSendDepthFrame)
         {
           sendImageSocket( clientsock ,networkDevice[0].depthFrame, networkDevice[0].depthWidth , networkDevice[0].depthHeight , networkDevice[0].depthChannels , networkDevice[0].depthBitsperpixel );
           networkDevice[0].okToSendDepthFrame=0;
         }
        // sendImageSocket(clientsock , char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel );
     //fprintf(stderr,"Serve Client looped\n");
     usleep(1000);
   }
*/


  int close_connection=0; // <- if this is set it means Serve Client must stop
 return 0;
}


int ammarserver_StopFrameServer(unsigned int devID)
{
 return 0;
}

#endif // USE_AMMARSERVER
