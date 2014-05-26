//gcc testquasirandom.c libQuasirandomness.a -o test

#include <stdio.h>
#include <stdlib.h>
#include "quasirandomness.h" 

int main(int argc, char *argv[])
{
  struct quasiRandomizerContext qrc;
  initializeQuasirandomnessContext(&qrc,640,480,0);

  float x,y,z;
  unsigned int i=0;
  for (i=0; i<100; i++)
  {
   getNextRandomPoint(&qrc,&x,&y,&z);    
   printf("Random Sample %u is %f %f %f\n",i,x,y,z); 
  } 



}
