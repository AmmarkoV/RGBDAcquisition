#ifndef DAEMON_H_INCLUDED
#define DAEMON_H_INCLUDED

int UpdateFrameServerImages(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);

int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort);
int StopFrameServer(unsigned int devID);

#endif // DAEMON_H_INCLUDED
