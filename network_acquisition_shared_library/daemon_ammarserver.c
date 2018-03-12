#include "daemon_ammarserver.h"


#include <stdio.h>
#include <stdlib.h>

#include "../tools/Codecs/codecs.h"
#include "../tools/Codecs/jpgInput.h"

#include "NetworkAcquisition.h"


#define MAX_CLIENTS_LISTENING_FOR 100





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
}





int ammarserver_StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{

}


int ammarserver_StopFrameServer(unsigned int devID)
{

}

