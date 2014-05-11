
#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include <stdio.h>
#include <stdlib.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>

#include "scene.h"
#include "glx.h"

static int snglBuf[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, None};
static int dblBuf[]  = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};

Display   *dpy;
Window     win;

GLboolean  doubleBuffer = GL_TRUE;

void fatalError(char *message)
{
  fprintf(stderr, "main: %s\n", message);
  exit(1);
}


int start_glx_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc, char **argv)
{

  XVisualInfo         *vi;
  Colormap             cmap;
  XSetWindowAttributes swa;
  GLXContext           cx;
  int                  dummy;


  /*** (1) open a connection to the X server ***/

  dpy = XOpenDisplay(NULL);
  if (dpy == NULL)
    fatalError("could not open display");

  /*** (2) make sure OpenGL's GLX extension supported ***/

  if(!glXQueryExtension(dpy, &dummy, &dummy))
    fatalError("X server has no OpenGL GLX extension");

  /*** (3) find an appropriate visual ***/

  /* find an OpenGL-capable RGB visual with depth buffer */
  vi = glXChooseVisual(dpy, DefaultScreen(dpy), dblBuf);
  if (vi == NULL)
  {
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), snglBuf);
    if (vi == NULL) fatalError("no RGB visual with depth buffer");
    doubleBuffer = GL_FALSE;
  }
  if(vi->class != TrueColor)
    fatalError("TrueColor visual required for this program");

  /*** (4) create an OpenGL rendering context  ***/

  /* create an OpenGL rendering context */
  cx = glXCreateContext(dpy, vi, /* no shared dlists */ None,
                        /* direct rendering if possible */ GL_TRUE);
  if (cx == NULL)
    fatalError("could not create rendering context");

  /*** (5) create an X window with the selected visual ***/

  /* create an X colormap since probably not using default visual */
  cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
  swa.colormap = cmap;
  swa.border_pixel = 0;
  swa.event_mask = KeyPressMask    | ExposureMask
                 | ButtonPressMask | StructureNotifyMask;
  win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0,
                      WIDTH, HEIGHT, 0, vi->depth, InputOutput, vi->visual,
                      CWBorderPixel | CWColormap | CWEventMask, &swa);
  XSetStandardProperties(dpy, win, "OpenGL Control Window", "main", None, argv, argc, NULL);

  /*** (6) bind the rendering context to the window ***/

  glXMakeCurrent(dpy, win, cx);

  /*** (7) request the X window to be displayed on the screen ***/

  if (viewWindow) { XMapWindow(dpy, win); }


  /*** (9) dispatch X events ***/
  return 1;
}



int glx_endRedraw()
{
  if (doubleBuffer) glXSwapBuffers(dpy, win);/* buffer swap does implicit glFlush */
  else glFlush();  /* explicit flush for single buffered case */
  return 1;
}


int glx_checkEvents()
{
  //GLboolean            needRedraw = GL_FALSE, recalcModelView = GL_TRUE;
  XEvent  event;
     while(XPending(dpy))
     {
      XNextEvent(dpy, &event);
      switch (event.type)
      {
        case KeyPress:
        {
          KeySym     keysym;
          XKeyEvent *kevent;
          char       buffer[1];
          // It is necessary to convert the keycode to a
          // keysym before checking if it is an escape */
          kevent = (XKeyEvent *) &event;
          if (   (XLookupString((XKeyEvent *)&event,buffer,1,&keysym,NULL) == 1)
              && (keysym == (KeySym)XK_Escape) )
            exit(0);


          handleUserInput(keysym,1,0,0);


          break;
        }
        case ButtonPress:
          switch (event.xbutton.button)
          {
            case 1: handleUserInput(1,1,0,0); break;
            case 2: handleUserInput(2,1,0,0); break;
            case 3: handleUserInput(3,1,0,0); break;
          }
          break;
        case ConfigureNotify:
          //glViewport(0, 0, event.xconfigure.width, event.xconfigure.height);
          windowSizeUpdated(event.xconfigure.width, event.xconfigure.height);
          /* fall through... */
        case Expose:
             #warning "redraws are not handled ?"
             // needRedraw=GL_TRUE;
          break;
      }
    }; /* loop to compress events */


         return 1;

     return 0;
}

