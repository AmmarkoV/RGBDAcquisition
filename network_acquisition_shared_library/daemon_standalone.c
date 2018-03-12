#include "daemon_standalone.h"

#include <stdio.h>
#include <stdlib.h>
#include "../tools/Codecs/codecs.h"
#include "../tools/Codecs/jpgInput.h"

#include "NetworkAcquisition.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/uio.h>


#define MAX_CLIENTS_LISTENING_FOR 100


struct serverState serverDevices[MAX_CLIENTS_LISTENING_FOR]={0};




void * standalone_ServeClient(void * ptr)
{
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

  while (serverDevices[0/*instanceID*/].serverRunning)
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



  int close_connection=0; // <- if this is set it means Serve Client must stop
}






int standalone_SpawnThreadToServeNewClient(unsigned int instanceID , int clientsock,struct sockaddr_in client,unsigned int clientlen)
{

  volatile struct PassToHTTPThread context={{0}};
  //memset((void*) &context,0,sizeof(struct PassToHTTPThread));

  context.keep_var_on_stack=1;

  context.clientsock=clientsock;
  context.client=client;
  context.clientlen=clientlen;
  context.pre_spawned_thread = 0; // THIS IS A !!!NEW!!! THREAD , NOT A PRESPAWNED ONE

  pthread_t server_thread_id;

  int retres = pthread_create(&server_thread_id,0/*&instance->attr*/,standalone_ServeClient,(void*) &context);

  while ( context.keep_var_on_stack !=2 ) { usleep(1000); fprintf(stderr,"."); }

  if (retres!=0) { retres = 0; } else { retres = 1; }


  return retres;
}







void * standalone_mainServerThread (void * ptr)
{
  struct passToServerThread * context = (struct passToServerThread *) ptr;
  if (context==0) { fprintf(stderr,"Error , mainServerThread called without a context\n");   return 0; }


  unsigned int serverlen = sizeof(struct sockaddr_in),clientlen = sizeof(struct sockaddr_in);
  struct sockaddr_in server;
  struct sockaddr_in client;

  int instanceID = context->id;
  context->doneWaiting=1;
  serverDevices[instanceID].serverRunning=0;
  serverDevices[instanceID].stopServer=1;

  int serversock = socket(AF_INET, SOCK_STREAM, 0);
    if ( serversock < 0 ) { fprintf(stderr,"Server Thread : Opening socket"); return 0; }
  serverDevices[instanceID].serversock = serversock;

  bzero(&client,clientlen);
  bzero(&server,serverlen);

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(serverDevices[instanceID].port);


  //We bind to our port..!
  if ( bind(serversock,(struct sockaddr *) &server,serverlen) < 0 )
    {
      fprintf(stderr,"Server Thread : Error binding master port!\nThe server may already be running ..\n");
      serverDevices[instanceID].serverRunning=0;
      return 0;
    }

  if ( listen(serversock,MAX_CLIENTS_LISTENING_FOR ) < 0 )  //Note that we are listening for a max number of clients as big as our maximum thread number..!
    {
      fprintf(stderr,"Server Thread : Failed to listen on server socket");
      serverDevices[instanceID].serverRunning=0;
      return 0;
    }


  serverDevices[instanceID].serverRunning=1;
  serverDevices[instanceID].stopServer=0;

  while ( (serverDevices[instanceID].serverRunning) && (serverDevices[instanceID].stopServer==0) )
  {
    fprintf(stderr,"\nServer Thread : Waiting for a new client @ port %u \n",serverDevices[instanceID].port);
    // Wait for client connection
    int clientsock=0;
    if ( (clientsock = accept(serversock,(struct sockaddr *) &client, &clientlen)) < 0) { error("Server Thread : Failed to accept client connection"); }
      else
      {
           fprintf(stderr,"Server Thread : Accepted new client , now deciding on prespawned vs freshly spawned.. \n");

            if (standalone_SpawnThreadToServeNewClient(instanceID,clientsock,client,clientlen))
            {
              // This request got served by a freshly spawned thread..!
              // Nothing to do here , proceeding to the next incoming connection..
              // if we failed then nothing can be done for this client
            } else
            {
                fprintf(stderr,"Server Thread : We dont have enough resources to serve client\n");
                close(clientsock);
            }

      }
 }
  serverDevices[instanceID].serverRunning=0;
  serverDevices[instanceID].stopServer=1;
  fprintf(stderr,"Server Stopped..");
  pthread_exit(0);
  return 0;
}



int standalone_StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{
  volatile struct passToServerThread context={ 0 };

  context.id=0;
  serverDevices[0].port=1234;
  //Creating the main WebServer thread..
  //It will bind the ports and start receiving requests and pass them over to new and prespawned threads
   pthread_t server_thread_id;
   int retres = pthread_create( &server_thread_id , 0 , standalone_mainServerThread,(void*) &context);
   if (retres!=0) { fprintf(stderr,"Cannot create thread for server\n"); return 0; }

   fprintf(stderr,"Waiting for server thread creation\n");
   while (!context.doneWaiting)
   {
     usleep(1000);
     fprintf(stderr,".");
   }


   unsigned int waitTime = 0; unsigned int maxWaitTime = 100;
   fprintf(stderr,"\nWaiting for server to bind ports etc \n");
   while ( ( ! serverDevices[0].serverRunning ) && (waitTime<maxWaitTime) )
   {
     fprintf(stderr,".");
     usleep(1000*1000);
     ++waitTime;
   }
   fprintf(stderr,"\n");


  return serverDevices[0].serverRunning;
}


int standalone_StopFrameServer(unsigned int devID)
{
 return 0;
}

