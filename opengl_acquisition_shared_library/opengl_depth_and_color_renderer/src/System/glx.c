
#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include <stdio.h>
#include <stdlib.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>

#include "glx.h"


int start_glx_stuff(int WIDTH,int HEIGHT,int openGLVersion,int viewWindow,int argc, char **argv)
{
  switch (openGLVersion)
  {
    case 1:
    case 2:
      return start_glx2_stuff(WIDTH,HEIGHT,viewWindow,argc,argv);
    break;

    case 3:
    case 4:
      return start_glx3_stuff(WIDTH,HEIGHT,viewWindow,argc,argv);
    break;
  };
  return 0;
}



int glx_endRedraw(int openGLVersion)
{
  switch (openGLVersion)
  {
    case 1:
    case 2:
      return glx2_endRedraw();
    break;

    case 3:
    case 4:
      return glx3_endRedraw();
    break;
  };
  return 1;
}


int glx_checkEvents(int openGLVersion)
{
  switch (openGLVersion)
  {
    case 1:
    case 2:
      return glx2_checkEvents();
    break;

    case 3:
    case 4:
      return glx3_checkEvents();
    break;
  };
     return 0;
}

