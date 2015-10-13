#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "locationService.h"

int main(int argc, char *argv[])
{
 fprintf(stderr,"Starting Location Service Tester!\n");

 if ( startLocationServices() )
 {
   unsigned int sampleNumber=0;
   while (1)
   {
   if ( pollLocationServices())
   {
     fprintf(stderr,"Sample %u => %0.2f %0.2f \n",sampleNumber++,getLat(),getLon());

   }
     usleep(10000);
     fprintf(stderr,".");
   }
   stopLocationServices();
 }

 fprintf(stderr,"Done..\n\n");
 return 0;
}
