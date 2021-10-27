#include "tools.h"

#include <sys/time.h>
#include <time.h>


unsigned long tickBaseAmmClient = 0;

unsigned long AmmClient_GetTickCountMicrosecondsInternal()
{
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBaseAmmClient==0)
   {
     tickBaseAmmClient = ts.tv_sec*1000000 + ts.tv_nsec/1000;
     return 0;
   }

   return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseAmmClient;
}

unsigned long AmmClient_GetTickCountMillisecondsInternal()
{
   //This returns a monotnic "uptime" value in milliseconds , it behaves like windows GetTickCount() but its not the same..
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBaseAmmClient==0)
   {
     tickBaseAmmClient = ts.tv_sec*1000 + ts.tv_nsec/1000000;
     return 0;
   }

   return ( ts.tv_sec*1000 + ts.tv_nsec/1000000 ) - tickBaseAmmClient;
}

