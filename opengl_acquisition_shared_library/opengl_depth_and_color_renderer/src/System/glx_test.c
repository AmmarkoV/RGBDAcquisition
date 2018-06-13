#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "glx3.h"


int windowSizeUpdated(unsigned int newWidth , unsigned int newHeight)
{
}

int handleUserInput(char key,int state,unsigned int x, unsigned int y)
{
}

int main(int argc, char **argv)
{
  int WIDTH=640;
  int HEIGHT=480;
  start_glx3_stuff(WIDTH,HEIGHT,1,argc,argv);


  while (1)
   {
     
     sleep(1);
     glx3_endRedraw();
   }
  
  stop_glx3_stuff();
 return 0;
}
