/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"

int readFromArg=0;
int photoShootOBJ=0;
float angleX=0.0,angleY=0.0,angleZ=0.0;
unsigned int width=640;
unsigned int height=480;

int main(int argc, char **argv)
{

  double * rodriguez = (double*) malloc(sizeof(double) * 3 );
  double * translation = (double*) malloc(sizeof(double) * 3 );
  double * camera = (double*) malloc(sizeof(double) * 9 );



 //Internal calibration
 camera[0]=535.784106;   camera[1]=0.0;         camera[2]=312.428312;
 camera[3]=0.0;          camera[4]=534.223354;  camera[5]=243.889369;
 camera[6]=0.0;          camera[7]=0.0;         camera[8]=1.0;


 #define USE_TEST 0

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
 #elif USE_TEST == 4
 //Test Calib
  translation[0]=-0.041674;  translation[1]=-0.076036; translation[2]=2.355294;
  rodriguez[0]=-0.733111;    rodriguez[1]=0.155142;    rodriguez[2]=0.172950;
 #elif USE_TEST == 5
  // ARTest
  translation[0]=-0.413148;  translation[1]=0.208111; translation[2]=1.987205;
  rodriguez[0]=2.052258;    rodriguez[1]=0.642598;    rodriguez[2]=-0.246993;
 #endif // USE_TEST


 #if USE_TEST == 0
  setOpenGLNearFarPlanes(1,5000);
  setOpenGLIntrinsicCalibration( (double*) camera);
  //setOpenGLExtrinsicCalibration( (double*) rodriguez, (double*) translation );
 #else
  setOpenGLNearFarPlanes(0.1,100);
  setOpenGLIntrinsicCalibration( (double*) camera);
  setOpenGLExtrinsicCalibration( (double*) rodriguez, (double*) translation );
 #endif



  int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-size")==0) {
                                        if (i+2<argc)
                                        {
                                         width=atof(argv[i+1]);
                                         height=atof(argv[i+2]);
                                        }
                                     } else
    if (strcmp(argv[i],"-photo")==0) {
                                        if (i+4<argc)
                                        {
                                         photoShootOBJ=atoi(argv[i+1]);
                                         angleX=atof(argv[i+2]);
                                         angleY=atof(argv[i+3]);
                                         angleZ=atof(argv[i+4]);
                                        }
                                     } else
    if (strcmp(argv[i],"-from")==0) {
                                        if (i+1<argc)
                                          readFromArg = i+1 ;
                                    }
  }

 if (readFromArg!=0) {   startOGLRendererSandbox(width,height,argv[readFromArg]); } else
                     {   startOGLRendererSandbox(width,height,0); /*0 defaults to scene.conf*/ }


  if (photoShootOBJ)
   {
     //fprintf(stderr,"Making a photoshoot of object %u",photoShootOBJ);
     snapOGLRendererPhotoshootSandbox(photoShootOBJ,angleX,angleY,angleZ);
     writeOpenGLColor("color.pnm",0,0,width,height);
     writeOpenGLDepth("depth.pnm",0,0,width,height);
     return 0;
   }


  snapOGLRendererSandbox(); // Snap a frame
  writeOpenGLColor("color.pnm",0,0,width,height);
  writeOpenGLDepth("depth.pnm",0,0,width,height);

   while (1)
    {
      snapOGLRendererSandbox();
    }


  free(rodriguez);
  free(translation);
  free(camera);

  stopOGLRendererSandbox();
  return 0;
}
