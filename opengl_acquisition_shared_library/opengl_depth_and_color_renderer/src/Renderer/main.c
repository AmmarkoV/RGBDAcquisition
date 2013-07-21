/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"

int main(int argc, char **argv)
{


//  float rodriguez[3]={ 1.911447 , 0.000701 , -0.028548};
//  float translation[3]={ -0.062989 , 0.159865 , 0.703045 };

 //box0603 Calib
 //float translation[3]={  0.215793 , -0.137982 , 0.767494 } ;
 //float rodriguez[3]={  0.029210 , -2.776582 , 1.451629 };



 float camera[9]=
 {  535.784106 , 0.0        , 312.428312 ,
    0.0        , 534.223354 , 243.889369 ,
    0.0        , 0.0        , 1.0 };

 //boxNew Calib
 float translation[3]={ -0.062989 ,0.159865 , 0.703045 } ;
 float rodriguez[3]={  1.911447 , 0.000701 , -0.028548 };


 //setOpenGLIntrinsicCalibration(camera);
 //setOpenGLExtrinsicCalibration( (float*) rodriguez, (float*) translation );


  startOGLRendererSandbox();

   while (1)
    {
      snapOGLRendererSandbox();
    }

  stopOGLRendererSandbox();
  return 0;
}
