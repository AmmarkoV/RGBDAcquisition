#ifndef DAEMON_H_INCLUDED
#define DAEMON_H_INCLUDED


int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort);
int StopFrameServer(unsigned int devID);

#endif // DAEMON_H_INCLUDED
