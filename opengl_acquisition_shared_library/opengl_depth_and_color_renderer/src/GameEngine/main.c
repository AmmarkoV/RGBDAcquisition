/** @file main.c
 *  @brief  A minimal binary that renders scene files using OGLRendererSandbox s
 *          X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11 -lpng -ljpeg
 *          X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11 -lpng -ljpeg
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

#include "../System/glx.h"
#include "../Scene/scene.h"



int main(int argc, char **argv)
{
    int width=640; int height=480; int framerate = 30;

    int readFromArg =0;

    int i=0;
    for (i=0; i<argc; i++)
    {
       if (strcmp(argv[i],"--from")==0)
        {
            if (i+1<argc)
            {
                readFromArg = i+1 ;
            }
        }
    }



    int started = 0;
    if (readFromArg!=0)
    {
        started=startOGLRendererSandbox(width,height,1 /*View OpenGL Window*/,argv[readFromArg]);
    }

    if (!started)
    {
        fprintf(stderr,"Could not start OpenGL Renderer Sandbox , please see log to find the exact reason of failure \n");
        return 0;
    }


    setKeyboardControl(1);

    struct VirtualStream * scene = getLoadedScene();

    unsigned int snappedFrames=0;
    while (1)
    {
        snapOGLRendererSandbox(framerate);

        scene->cameraPose.posX+=1;
        scene->cameraUserDelta.posX+=1;


       ++snappedFrames;
    }


    stopOGLRendererSandbox();
    return 0;
}
