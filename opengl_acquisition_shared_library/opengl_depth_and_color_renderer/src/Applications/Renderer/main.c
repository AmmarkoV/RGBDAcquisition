/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
 *  @author Ammar Qammaz (AmmarkoV)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../Library/OGLRendererSandbox.h"
#include "../../Library/Scene/scene.h"


#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include "../../Library/System/glx.h"


int readFromArg=0,writeToArg=0,doFileOutput=0;
int photoShootOBJ=0;
unsigned int maxFrames=0;
float angleX=0.0,angleY=0.0,angleZ=0.0;
unsigned int width=640;
unsigned int height=480;
unsigned int forceKeyboardControl=0;

unsigned int columns=22,rows=21;
float distance = 1.5;

int framerate=30;

static int myMkdir(const char * prefix,const char * dirname)
{
    char filename[FILENAME_MAX]= {0};
    // - - - - - - - - - - - - - - - -
    if ( (prefix==0)||(dirname==0) ) { return 0; } else
    if ( (prefix==0)&&(dirname!=0) ) { snprintf(filename,FILENAME_MAX,"mkdir -p %s",dirname); } else
    if ( (prefix!=0)&&(dirname==0) ) { snprintf(filename,FILENAME_MAX,"mkdir -p %s",prefix); } else
                                     { snprintf(filename,FILENAME_MAX,"mkdir -p %s/%s",prefix,dirname); }

    int i=system(filename);
    if (i!=0)
    {
        fprintf(stderr,"Could not create directory %s/%s.. \n",prefix,dirname);
        return 0;
    }
    return 1;
}




int reallyFastCheckForLinuxGPUWithoutPBuffer()
{
   FILE * fp = popen("glxinfo | grep NVIDIA", "r");
   if (fp == 0 ) { return 1; }

 /* Read the output a line at a time - output it. */

  char what2GetBack[513]={0};

  fgets(what2GetBack,512, fp);
  /* close */
  pclose(fp);

  fprintf(stderr,"reallyFastCheckForLinuxGPUWithoutPBuffer = %s \n",what2GetBack);

  if (strlen(what2GetBack)<1) { return 1; }

  return 0;
}




int main(int argc,const char **argv)
{

    float * rodriguez = (float*) malloc(sizeof(float) * 3 );
    float * translation = (float*) malloc(sizeof(float) * 3 );
    float * camera = (float*) malloc(sizeof(float) * 9 );
    float scaleToDepthUnit = 1.0;



//Internal calibration
    camera[0]=535.784106;
    camera[1]=0.0;
    camera[2]=312.428312;
    camera[3]=0.0;
    camera[4]=534.223354;
    camera[5]=243.889369;
    camera[6]=0.0;
    camera[7]=0.0;
    camera[8]=1.0;

    translation[0]=0.0;
    translation[1]=0.0;
    translation[2]=0.0;
    rodriguez[0]=0.0;
    rodriguez[1]=0.0;
    rodriguez[2]=0.0;

    setOpenGLNearFarPlanes(1,15000);
    unsigned int viewWindow=1;

    int i=0;
    for (i=0; i<argc; i++)
    {



        if (strcmp(argv[i],"--asap")==0)
        {
            framerate=0;
        } else
        if (strcmp(argv[i],"-from")==0)
        {
            fprintf(stderr,"Parameter Syntax is --from NOT -from ..!! \n");
            exit(0);
        } else
        if (strcmp(argv[i],"--test")==0)
        {
            internalTest();
            exit(0);
        }
        else if (strcmp(argv[i],"--intrinsics")==0)
        {
            if (i+8<argc)
            {
                int z=0;
                for (z=0; z<9; z++)
                {
                    camera[z]=atof(argv[z+i+1]);
                }
                setOpenGLIntrinsicCalibration( (float*) camera);
            }
        }
        else if (strcmp(argv[i],"--extrinsics")==0)
        {
            if (i+7<argc)
            {
                translation[0]=atof(argv[i+1]);
                translation[1]=atof(argv[i+2]);
                translation[2]=atof(argv[i+3]);
                rodriguez[0]=atof(argv[i+4]);
                rodriguez[1]=atof(argv[i+5]);
                rodriguez[2]=atof(argv[i+6]);
                scaleToDepthUnit = atof(argv[i+7]);
                setOpenGLExtrinsicCalibration( (float*) rodriguez, (float*) translation , scaleToDepthUnit);
            }
        }
        else if ( (strcmp(argv[i],"--resolution")==0) ||
                  (strcmp(argv[i],"--size")==0) )
        {
            if (i+2<argc)
            {
                width=atof(argv[i+1]);
                height=atof(argv[i+2]);
                fprintf(stderr,"Resolution selected is (%ux%u)\n",width,height);
            }
        }
        else if (
            (strcmp(argv[i],"--photo")==0) ||
            (strcmp(argv[i],"--photoshoot")==0)
        )
        {
            if (i+4<argc)
            {
                viewWindow=reallyFastCheckForLinuxGPUWithoutPBuffer();
                photoShootOBJ=atoi(argv[i+1]);
                angleX=atof(argv[i+2]);
                angleY=atof(argv[i+3]);
                angleZ=atof(argv[i+4]);
                columns=atoi(argv[i+5]);
                rows=atoi(argv[i+6]);
            }
        }
        else if (strcmp(argv[i],"--from")==0)
        {
            if (i+1<argc)
            {
                readFromArg = i+1 ;
            }
        }
        else if (strcmp(argv[i],"--to")==0)
        {
            if (i+1<argc)
            {
                writeToArg = i+1 ;
                doFileOutput=1;
                fprintf(stderr,"Will write data to %s\n",argv[writeToArg]);
            }
        }
        else if (strcmp(argv[i],"--shader")==0)
        {
            //char * vertShaderFilename , char * fragShaderFilename );
            enableShaders(argv[i+1],argv[i+2]);
        }
        else if (strcmp(argv[i],"--keyboard")==0)
        {
            forceKeyboardControl=1;
        }
        else if (strcmp(argv[i],"--maxFrames")==0)
        {
            maxFrames=atoi(argv[i+1]);
        }
    }



    int started = 0;
    if (readFromArg!=0)
    {
        started=startOGLRendererSandbox(argc,argv,width,height,viewWindow /*View OpenGL Window*/,argv[readFromArg]);
    }
    else
    {
        started=startOGLRendererSandbox(argc,argv,width,height,viewWindow /*View OpenGL Window*/,0); /*0 defaults to scene.conf*/
    }


    if (!started)
    {
        fprintf(stderr,"Could not start OpenGL Renderer Sandbox , please see log to find the exact reason of failure \n");
        return 0;
    }

    if (photoShootOBJ)
    {
        float angXVariance=360,angYVariance=360,angZVariance=360;
        fprintf(stderr,"Making a photoshoot of object %d with a size %ux%u image output\n",photoShootOBJ,width,height);

        void * oglPhotoShoot = createOGLRendererPhotoshootSandbox(
                                                                   (void*) getLoadedScene(),
                                                                   (void*) getLoadedModelStorage(),
                                                                   photoShootOBJ,
                                                                   columns,rows,
                                                                   distance,
                                                                   angleX,angleY,angleZ,
                                                                   angXVariance,angYVariance,angZVariance
                                                                  );

        snapOGLRendererPhotoshootSandbox(oglPhotoShoot , photoShootOBJ,columns,rows,distance,angleX,angleY,angleZ,angXVariance,angYVariance,angZVariance);
        fprintf(stderr,"Writing photoshoot output in a file with size %ux%u\n",width,height);
        writeOpenGLColor("color.pnm",0,0,width,height);
        writeOpenGLDepth("depth.pnm",0,0,width,height);

        destroyOGLRendererPhotoshootSandbox( oglPhotoShoot );
        return 0;
    }

    if ( forceKeyboardControl )
    {
        setKeyboardControl(1);
    }

    char filename[FILENAME_MAX]= {0};
    if (doFileOutput)
    {
        myMkdir("frames",argv[writeToArg]);
    }



    unsigned int snappedFrames=0;
    while (1)
    {
        snapOGLRendererSandbox(framerate);


        if (doFileOutput)
        {
            snprintf(filename,FILENAME_MAX,"frames/%s/colorFrame_0_%05u.pnm",argv[writeToArg],snappedFrames);
            writeOpenGLColor(filename,0,0,width,height);


            snprintf(filename,FILENAME_MAX,"frames/%s/depthFrame_0_%05u.pnm",argv[writeToArg],snappedFrames);
            writeOpenGLDepth(filename,0,0,width,height);

        }


        if (maxFrames!=0)
        {
          if (maxFrames<=snappedFrames)
          {
            fprintf(stderr,"Reached target of %u frames , stopping\n",maxFrames);
            break;
          }
        }

       ++snappedFrames;
    }


    free(rodriguez);
    free(translation);
    free(camera);

    stopOGLRendererSandbox();
    return 0;
}
