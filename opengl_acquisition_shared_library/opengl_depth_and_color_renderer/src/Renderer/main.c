/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"

int main(int argc, char **argv)
{
  startOGLRendererSandbox();


  float rodriguez[3]={ 1.911447 , 0.000701 , -0.028548};
  float translation[3]={ -0.062989 , 0.159865 , 0.703045 };

  setOpenGLExtrinsicCalibration( (float*) rodriguez, (float*) translation );

   while (1)
    {
      snapOGLRendererSandbox();
    }

  stopOGLRendererSandbox();
  return 0;
}
