#include "daemon.h"

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


struct serverState
{
   int sock;
   int serversock;

   int port;

   int serverThreadId;
   int serverRunning;
   int stopServer;
};

struct passToServerThread
{
   int id;
};


struct serverState serverDevices[MAX_CLIENTS_LISTENING_FOR]={0};




void * mainServerThread (void * ptr)
{
  struct passToServerThread * context = (struct passToServerThread *) ptr;
  if (context==0) { fprintf(stderr,"Error , mainServerThread called without a context\n");   return 0; }


  unsigned int serverlen = sizeof(struct sockaddr_in),clientlen = sizeof(struct sockaddr_in);
  struct sockaddr_in server;
  struct sockaddr_in client;

  int instanceID = context->id;

  int serversock = socket(AF_INET, SOCK_STREAM, 0);
    if ( serversock < 0 ) { fprintf(stderr,"Server Thread : Opening socket"); return 0; }
  serverDevices[instanceID].serversock = serversock;

  bzero(&client,clientlen);
  bzero(&server,serverlen);

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = htons(serverDevices[instanceID].port);
/*

  //We bind to our port..!
  if ( bind(serversock,(struct sockaddr *) &server,serverlen) < 0 )
    {
      fprintf(stderr,"Server Thread : Error binding master port!\nThe server may already be running ..\n");
      instance->server_running=0;
      return 0;
    }

  //MAX_CLIENT_THREADS <- this could also be used instead of MAX_CLIENTS_LISTENING_FOR
  //I am trying a larger listen queue to hold incoming connections regardless of the serving threads
  //so that they will be used later
  if ( listen(serversock,MAX_CLIENTS_LISTENING_FOR ) < 0 )  //Note that we are listening for a max number of clients as big as our maximum thread number..!
         {
           fprintf(stderr,"Server Thread : Failed to listen on server socket");
           instance->server_running=0;
           return 0;
         }



  //If we made it this far , it means we got ourselves the port we wanted and we can start serving requests , but before we do that..
  //The next call Pre"forks" a number of threads specified in configuration.h ( MAX_CLIENT_PRESPAWNED_THREADS )
  //They can reduce latency by up tp 10ms on a Raspberry Pi , without any side effects..
  PreSpawnThreads(instance);

  while ( (instance->server_running) && (instance->stop_server==0) && (GLOBAL_KILL_SERVER_SWITCH==0) )
  {
    fprintf(stderr,"\nServer Thread : Waiting for a new client\n");
    // Wait for client connection
    int clientsock=0;
    if ( (clientsock = accept(serversock,(struct sockaddr *) &client, &clientlen)) < 0) { error("Server Thread : Failed to accept client connection"); }
      else
      {
           fprintf(stderr,"Server Thread : Accepted new client , now deciding on prespawned vs freshly spawned.. \n");


            if (SpawnThreadToServeNewClient(instance,clientsock,client,clientlen,instance->webserver_root,instance->templates_root))
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
  instance->server_running=0;
  instance->stop_server=2;

  fprintf(stderr,"Server Stopped..");
  //It should already be closed so skipping this : close(serversock);
  pthread_exit(0);*/
  return 0;
}










int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{
  volatile struct passToServerThread context={ 0 };
  //Creating the main WebServer thread..
  //It will bind the ports and start receiving requests and pass them over to new and prespawned threads
   pthread_t server_thread_id;
   int retres = pthread_create( &server_thread_id , 0 , mainServerThread,(void*) &context);
  // instance->server_thread_id = server_thread_id;

 return retres;
}


int StopFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{

 return 0;
}

