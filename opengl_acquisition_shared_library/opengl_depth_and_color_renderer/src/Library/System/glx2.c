
#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include <stdio.h>
#include <stdlib.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>

#include "../Scene/scene.h"
#include "glx2.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


static int snglBuf[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, None};
static int dblBuf[]  = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};

Display   *dpy;
Window     win2;

GLboolean  doubleBuffer = GL_TRUE;

void fatalError(char *message)
{
  fprintf(stderr, RED "fatal Error: %s\n" NORMAL, message);
  //exit(1);
}

static Bool WaitForNotify( Display *dpy, XEvent *event, XPointer arg )
{
    return (event->type == MapNotify) && (event->xmap.window == (Window) arg);
}


int start_glx2_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc,const char **argv)
{
  fprintf(stderr,"start_glx2_stuff\n");
  XVisualInfo         *vi;
  Colormap             cmap;
  XSetWindowAttributes swa;
  GLXContext           cx;
  int                  dummy;

  int debugMessages=0;


  if (debugMessages) { fprintf(stderr,"(1) open a connection to the X server\n"); }
  /*** (1) open a connection to the X server ***/

  dpy = XOpenDisplay(NULL);
  if (dpy == NULL)
    {
     fatalError("Could not open display");
     return 0;
    }

  if (debugMessages) { fprintf(stderr,"(2) make sure OpenGL's GLX extension supported\n"); }
  /*** (2) make sure OpenGL's GLX extension supported ***/

  if(!glXQueryExtension(dpy, &dummy, &dummy))
    {
      fatalError("X server has no OpenGL GLX extension");
      return 0;
    }

  //viewWindow=1; //FORCE

  //Two seperate glx initialization procedures wether we want a window or offscreen rendering..
  if (viewWindow)
    {

  if (debugMessages) { fprintf(stderr,"(3) find an appropriate visual .. "); }
  /*** (3) find an appropriate visual ***/
  /* find an OpenGL-capable RGB visual with depth buffer */
  vi = glXChooseVisual(dpy, DefaultScreen(dpy), dblBuf);
  if (debugMessages) { fprintf(stderr," survived \n"); }
  if (vi == NULL)
  {
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), snglBuf);
    if (vi == NULL)
    {
      fatalError("no RGB visual with depth buffer");
      return 0;
    }
    doubleBuffer = GL_FALSE;
  }
  if(vi->class != TrueColor)
  {
    fatalError("TrueColor visual required for this program");
    return 0;
  }

  if (debugMessages) { fprintf(stderr,"(4) create an OpenGL rendering context\n"); }
  /*** (4) create an OpenGL rendering context  ***/

  /* create an OpenGL rendering context */
  cx = glXCreateContext(dpy,
                        vi,
                        /* no shared dlists */ None,
                        /* direct rendering if possible */ GL_TRUE
                        );
  if (cx == NULL)
    {
      fatalError("could not create rendering context");
      return 0;
    }

  if ( ! glXIsDirect ( dpy, cx ) ) { printf( "Indirect GLX rendering context obtained\n" ); } else
                                   { printf( "Direct GLX rendering context obtained\n" );   }


  if (debugMessages) { fprintf(stderr,"(5) create an X window with the selected visual\n"); }
  /*** (5) create an X window with the selected visual ***/

  /* create an X colormap since probably not using default visual */
  cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
  swa.colormap = cmap;
  swa.border_pixel = 0;
  swa.event_mask = KeyPressMask    | ExposureMask
                 | ButtonPressMask | StructureNotifyMask;



        win2 = XCreateWindow(dpy,
                      RootWindow(dpy, vi->screen), 0, 0,
                      WIDTH, HEIGHT,
                      0, vi->depth,
                      InputOutput,
                      vi->visual,
                      CWBorderPixel | CWColormap | CWEventMask, &swa);
       XSetStandardProperties(dpy, win2, "OpenGL2.x Control Window", "main", None, argv, argc, NULL);

       if (debugMessages) { fprintf(stderr,"(6) bind the rendering context to the window \n"); }
       /*** (6) bind the rendering context to the window ***/


       glXMakeCurrent(dpy, win2, cx);
       if (debugMessages) { fprintf(stderr,"(7) request the X window to be displayed on the screen\n"); }
       /*** (7) request the X window to be displayed on the screen ***/
      //Request the window to get Displayed
      XMapWindow(dpy, win2);
      //Wait for window to be visible
      XEvent event;
      XIfEvent( dpy, &event, WaitForNotify, (XPointer) win2 );
    } else
    {
      fprintf(stderr,"Will not display a window..\n");

      //static int visualAttribs[] = { None };
      int w=WIDTH, h=HEIGHT;
      int visualAttribs[]={
                           GLX_RENDER_TYPE, GLX_RGBA_BIT,
                           GLX_MAX_PBUFFER_WIDTH, w,
                           GLX_MAX_PBUFFER_HEIGHT, h,
                           GLX_RED_SIZE, 4,
                           GLX_GREEN_SIZE, 4,
                           GLX_BLUE_SIZE, 4,
                           GLX_DRAWABLE_TYPE,GLX_PBUFFER_BIT,
                           GLX_DEPTH_SIZE, 24,
                           None
                           };

      int numberOfFramebufferConfigurations = 0;
      fprintf(stderr,"glXChooseFBConfig\n");
      GLXFBConfig* fbConfigs = glXChooseFBConfig( dpy, DefaultScreen(dpy), visualAttribs, &numberOfFramebufferConfigurations );
      if ( (fbConfigs == NULL) || (numberOfFramebufferConfigurations <= 0) )
        {
            fatalError("P-Buffers not supported.\n");
            return 0;
        }


     int pbufferAttribs[]={
                           GLX_PBUFFER_WIDTH, w,
                           GLX_PBUFFER_HEIGHT, h,
                           GLX_NONE
                          };

      fprintf(stderr,"glXCreatePbuffer\n");
      GLXPbuffer pbuffer = glXCreatePbuffer( dpy,fbConfigs[0], pbufferAttribs );
      if (pbuffer==0)
        {
            fatalError("glXCreatePbuffer failed..\n");
            return 0;
        }

        cx = glXCreateNewContext(
                                 dpy,
                                 fbConfigs[0],
                                 GLX_RGBA_TYPE,
                                 NULL,
                                 GL_TRUE
                                 );
        if (!cx)
        {
         fatalError("Failed to create graphics context.\n");
         return 0;
        }


      // clean up:
      XFree( fbConfigs );
      XSync( dpy, False );

      fprintf(stderr,"glXMakeContextCurrent\n");
      if ( !glXMakeContextCurrent( dpy, pbuffer, pbuffer, cx ) )
      {
        fatalError("Could not start rendering to pbuffer fbo");
        return 0;
      }


      //glXMakeContextCurrent(dpy,None,None,cx);
    }


  /*** (9) dispatch X events ***/
  return 1;
}



int glx2_endRedraw()
{
  if (doubleBuffer) glXSwapBuffers(dpy, win2);/* buffer swap does implicit glFlush */
  else glFlush();  /* explicit flush for single buffered case */
  return 1;
}


int glx2_checkEvents()
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
        case ButtonRelease:
        case ButtonPress:
          switch (event.xbutton.button)
          {
            case 1: handleUserInput(1,(event.type==ButtonPress),event.xmotion.x_root,event.xmotion.y_root); break;
            case 2: handleUserInput(2,(event.type==ButtonPress),event.xmotion.x_root,event.xmotion.y_root); break;
            case 3: handleUserInput(3,(event.type==ButtonPress),event.xmotion.x_root,event.xmotion.y_root); break;
          }
          break;
        case ConfigureNotify:
          //glViewport(0, 0, event.xconfigure.width, event.xconfigure.height);
          fprintf(stderr,"Received window configuration event..\n");
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

