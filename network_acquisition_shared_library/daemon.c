#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>


#if USE_AMMARSERVER
 #include "daemon_ammarserver.h"
#endif // USE_AMMARSERVER


int UpdateFrameServerImages(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  #if USE_AMMARSERVER
    ammarserver_UpdateFrameServerImages(frameServerID,streamNumber,pixels,width,height,channels,bitsperpixel);
  #endif // USE_AMMARSERVER
  fprintf(stderr,"UpdateFrameServerImages: NetworkAcquisition backbone not implemented , please use the ENABLE_AMMARSERVER_BROADCAST in CMake \n");
  return 0;
}

int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{
  #if USE_AMMARSERVER
    ammarserver_StartFrameServer(devID ,bindAddr,bindPort);
  #endif // USE_AMMARSERVER
  fprintf(stderr,"StartFrameServer: NetworkAcquisition backbone not implemented , please use the ENABLE_AMMARSERVER_BROADCAST in CMake \n");
  return 0;
}


int StopFrameServer(unsigned int devID)
{
  #if USE_AMMARSERVER
    ammarserver_StopFrameServer(devID);
  #endif // USE_AMMARSERVER
  fprintf(stderr,"StopFrameServer: NetworkAcquisition backbone not implemented , please use the ENABLE_AMMARSERVER_BROADCAST in CMake \n");
 return 0;
}

