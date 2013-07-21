/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"

int main(int argc, char **argv)
{

  double * rodriguez = (double*) malloc(sizeof(double) * 3 );
  double * translation = (double*) malloc(sizeof(double) * 3 );
  double * camera = (double*) malloc(sizeof(double) * 9 );

 //box0603 Calib
 //float translation[3]={  0.215793 , -0.137982 , 0.767494 } ;
 //float rodriguez[3]={  0.029210 , -2.776582 , 1.451629 };


 //Internal calibration
 camera[0]=535.784106;   camera[1]=0.0;         camera[2]=312.428312;
 camera[3]=0.0;          camera[4]=534.223354;  camera[5]=243.889369;
 camera[6]=0.0;          camera[7]=0.0;         camera[8]=1.0;

 //boxNew Calib
 translation[0]=-0.062989;  translation[1]=0.159865; translation[2]=0.703045;
 rodriguez[0]=1.911447;     rodriguez[1]=0.000701;   rodriguez[2]=-0.028548;


 setOpenGLIntrinsicCalibration( (double*) camera);
 //setOpenGLExtrinsicCalibration( (double*) rodriguez, (double*) translation );


  startOGLRendererSandbox();

   while (1)
    {
      snapOGLRendererSandbox();
    }


  free(rodriguez);
  free(translation);

  stopOGLRendererSandbox();
  return 0;
}
