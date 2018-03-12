#include "daemon.h"

#include <stdio.h>
#include <stdlib.h>

int StartFrameServer(unsigned int devID , char * bindAddr , int bindPort)
{
  #if USE_AMMARSERVER

  #endif // USE_AMMARSERVER
  fprintf(stderr,"StartFrameServer: NetworkAcquisition backbone not implemented , please use the ENABLE_AMMARSERVER_BROADCAST in CMake \n");
  return 0;
}


int StopFrameServer(unsigned int devID)
{
  #if USE_AMMARSERVER

  #endif // USE_AMMARSERVER
  fprintf(stderr,"StopFrameServer: NetworkAcquisition backbone not implemented , please use the ENABLE_AMMARSERVER_BROADCAST in CMake \n");
  return 0;
 return 0;
}

