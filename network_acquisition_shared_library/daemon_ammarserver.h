#ifndef DAEMON_AMMARSERVER_H_INCLUDED
#define DAEMON_AMMARSERVER_H_INCLUDED

int ammarserver_StartFrameServer(unsigned int devID , char * bindAddr , int bindPort);
int ammarserver_StopFrameServer(unsigned int devID);

#endif // DAEMON_H_INCLUDED
