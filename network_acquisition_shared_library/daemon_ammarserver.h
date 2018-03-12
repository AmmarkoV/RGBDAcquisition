#ifndef DAEMON_AMMARSERVER_H_INCLUDED
#define DAEMON_AMMARSERVER_H_INCLUDED

int ammarserver_UpdateFrameServerImages(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);
int ammarserver_StartFrameServer(unsigned int devID , char * bindAddr , int bindPort);
int ammarserver_StopFrameServer(unsigned int devID);

#endif // DAEMON_H_INCLUDED
