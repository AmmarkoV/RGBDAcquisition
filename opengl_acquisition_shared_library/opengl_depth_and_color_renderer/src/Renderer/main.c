/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"


#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include "../glx.h"


int readFromArg=0;
int photoShootOBJ=0;
float angleX=0.0,angleY=0.0,angleZ=0.0;
unsigned int width=640;
unsigned int height=480;

unsigned int columns=22,rows=21;
float distance = 30;

int main(int argc, char **argv)
{

  double * rodriguez = (double*) malloc(sizeof(double) * 3 );
  double * translation = (double*) malloc(sizeof(double) * 3 );
  double * camera = (double*) malloc(sizeof(double) * 9 );
  double scaleToDepthUnit = 1.0;



 //Internal calibration
 camera[0]=535.784106;   camera[1]=0.0;         camera[2]=312.428312;
 camera[3]=0.0;          camera[4]=534.223354;  camera[5]=243.889369;
 camera[6]=0.0;          camera[7]=0.0;         camera[8]=1.0;

  translation[0]=0.0;  translation[1]=0.0; translation[2]=0.0;
  rodriguez[0]=0.0;    rodriguez[1]=0.0;    rodriguez[2]=0.0;

  setOpenGLNearFarPlanes(1,15000);

  int i=0;
  for (i=0; i<argc; i++)
  {

    if (strcmp(argv[i],"-test")==0) { doTest(); exit(0); } else
    if (strcmp(argv[i],"-intrinsics")==0) {
                                           if (i+8<argc) {
                                                          int z=0;
                                                          for (z=0; z<9; z++) { camera[z]=atof(argv[z+i+1]); }
                                                          setOpenGLIntrinsicCalibration( (double*) camera);
                                                         }
                                          } else
    if (strcmp(argv[i],"-extrinsics")==0) {
                                           if (i+7<argc) {
                                                              translation[0]=atof(argv[i+1]);  translation[1]=atof(argv[i+2]); translation[2]=atof(argv[i+3]);
                                                              rodriguez[0]=atof(argv[i+4]);    rodriguez[1]=atof(argv[i+5]);    rodriguez[2]=atof(argv[i+6]);
                                                              scaleToDepthUnit = atof(argv[i+7]);
                                                              setOpenGLExtrinsicCalibration( (double*) rodriguez, (double*) translation , scaleToDepthUnit);
                                                         }
                                          } else
    if ( (strcmp(argv[i],"-resolution")==0) ||
         (strcmp(argv[i],"-size")==0) ){
                                        if (i+2<argc)
                                        {
                                         width=atof(argv[i+1]);
                                         height=atof(argv[i+2]);
                                        }
                                     } else
    if (
         (strcmp(argv[i],"-photo")==0) ||
         (strcmp(argv[i],"-photoshoot")==0)
        )
                                      {
                                        if (i+4<argc)
                                        {
                                         photoShootOBJ=atoi(argv[i+1]);
                                         angleX=atof(argv[i+2]);
                                         angleY=atof(argv[i+3]);
                                         angleZ=atof(argv[i+4]);
                                        }
                                      } else
    if (strcmp(argv[i],"-from")==0)   {
                                        if (i+1<argc)
                                          { readFromArg = i+1 ; }
                                      }  else
    if (strcmp(argv[i],"-shader")==0) {
                                       enableShaders(argv[i+1],argv[i+2]);
                                      }
  }


 int started = 0;
 if (readFromArg!=0) {   started=startOGLRendererSandbox(width,height,1 /*View OpenGL Window*/,argv[readFromArg]); } else
                     {   started=startOGLRendererSandbox(width,height,1 /*View OpenGL Window*/,0); /*0 defaults to scene.conf*/ }

 usleep(100);

 if (!started)
 {
    fprintf(stderr,"Could not start OpenGL Renderer Sandbox , please see log to find the exact reason of failure \n");
    return 0;
 }

  if (photoShootOBJ)
   {
     float angXVariance=60,angYVariance=60,angZVariance=30;
     //fprintf(stderr,"Making a photoshoot of object %u",photoShootOBJ);

     void * oglPhotoShoot = createOGLRendererPhotoshootSandbox( photoShootOBJ,columns,rows,distance,angleX,angleY,angleZ,angXVariance,angYVariance,angZVariance );

     snapOGLRendererPhotoshootSandbox(oglPhotoShoot , photoShootOBJ,columns,rows,distance,angleX,angleY,angleZ,angXVariance,angYVariance,angZVariance);
     writeOpenGLColor("color.pnm",0,0,width,height);
     writeOpenGLDepth("depth.pnm",0,0,width,height);

     destroyOGLRendererPhotoshootSandbox( oglPhotoShoot );
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
