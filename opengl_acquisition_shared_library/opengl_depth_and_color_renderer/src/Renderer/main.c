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



 //Internal calibration
 camera[0]=535.784106;   camera[1]=0.0;         camera[2]=312.428312;
 camera[3]=0.0;          camera[4]=534.223354;  camera[5]=243.889369;
 camera[6]=0.0;          camera[7]=0.0;         camera[8]=1.0;


 #define USE_TEST 3

 #if   USE_TEST == 0
  translation[0]=0.0;  translation[1]=0.0; translation[2]=0.0;
  rodriguez[0]=0.0;    rodriguez[1]=0.0;    rodriguez[2]=0.0;
 #elif USE_TEST == 1
  //box0603 Calib
  translation[0]=0.215793; translation[1]=-0.137982; translation[2]=0.767494;
  rodriguez[0]=0.029210; rodriguez[1]=-2.776582; rodriguez[2]=1.451629;
 #elif USE_TEST == 2
  //boxNew Calib
  translation[0]=-0.062989;  translation[1]=0.159865; translation[2]=0.703045;
  rodriguez[0]=1.911447;     rodriguez[1]=0.000701;   rodriguez[2]=-0.028548;
 #elif USE_TEST == 3
  //Test Calib
  translation[0]=0.056651;  translation[1]=-0.000811; translation[2]=0.601942;
  rodriguez[0]=0.829308;    rodriguez[1]=2.251753;    rodriguez[2]=-1.406462;
 #endif // USE_TEST


 setOpenGLIntrinsicCalibration( (double*) camera);
 setOpenGLExtrinsicCalibration( (double*) rodriguez, (double*) translation );


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
