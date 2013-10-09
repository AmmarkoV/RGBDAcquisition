#ifndef DAEMON_H_INCLUDED
#define DAEMON_H_INCLUDED

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/uio.h>

struct serverState
{
   int sock;
   int serversock;

   int port;

   int serverThreadId;
   int serverRunning;
   int stopServer;
};

struct PassToHTTPThread
{
     volatile int keep_var_on_stack;

     struct sockaddr_in client;
     unsigned int clientlen;
     unsigned int thread_id;
     unsigned int port;

     int clientsock;

   int id;
     int pre_spawned_thread;
};


struct passToServerThread
{
   int id;
   int doneWaiting;
};


int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort);
int StopFrameServer(unsigned int devID);

#endif // DAEMON_H_INCLUDED
