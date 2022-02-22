//From : https://www.khronos.org/opengl/wiki/Tutorial:_OpenGL_3.0_Context_Creation_(GLX)
//Compile using :  gcc -o gl3 glx3.c -lGL -lX11
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>



#include <GL/gl.h>
#include <GL/glx.h>

#include "glx3.h"

#include "../Scene/scene.h"

GLXDrawable whatToSwap=0;
GLboolean  doubleBufferGLX3 = GL_TRUE;
static int dblBuf[]  = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};

Display   *display;
Window     win;
GLXContext ctx = 0;
Colormap cmap;


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */


#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);



void fatalErrorGLX3(char *message)
{
  fprintf(stderr, RED "fatal Error: %s\n" NORMAL, message);
  //exit(1);
}


// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static int isExtensionSupported(const char *extList, const char *extension)
{
  const char *start;
  const char *where, *terminator;

  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if (where || *extension == '\0')
    return 0;

  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  for (start=extList;;) {
    where = strstr(start, extension);

    if (!where)
      break;

    terminator = where + strlen(extension);

    if ( where == start || *(where - 1) == ' ' )
      if ( *terminator == ' ' || *terminator == '\0' )
        return 1;

    start = terminator;
  }

  return 0;
}

static int ctxErrorOccurred = 0;
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev )
{
    ctxErrorOccurred = 1;
    return 0;
}




static void run_getenv (const char * name)
{
    char * value = getenv (name);
    if (! value) {
        printf ("'%s' is not set.\n", name);
    }
    else {
        printf ("%s = %s\n", name, value);
    }
}


int disableVSync() //This needs to be done before initialization of GLX3 stuff..
{
   //NVIDIA VSYNC ENVIRONMENT FLAG
   run_getenv("__GL_SYNC_TO_VBLANK");
   setenv ("__GL_SYNC_TO_VBLANK", "0", 0);
   run_getenv("__GL_SYNC_TO_VBLANK");

   //INTEL VSYNC ENVIRONMENT FLAG
   run_getenv("vblank_mode");
   setenv ("vblank_mode", "0", 0);
   run_getenv("vblank_mode");
}


int start_glx3_stuffWindowed(int WIDTH,int HEIGHT,int argc,const char **argv)
{
  fprintf(stderr,"start_glx3_stuffWindowed\n");
  doubleBufferGLX3 = GL_TRUE;

  display = XOpenDisplay(NULL);

  if (!display)
  {
    printf("Failed to open X display\n");
    return 0;
  }

  // Get a matching FB config
  static int visual_attribs[] =
    {
      GLX_X_RENDERABLE    , True,
      GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
      GLX_RENDER_TYPE     , GLX_RGBA_BIT,
      GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
      GLX_RED_SIZE        , 8,
      GLX_GREEN_SIZE      , 8,
      GLX_BLUE_SIZE       , 8,
      GLX_ALPHA_SIZE      , 8,
      GLX_DEPTH_SIZE      , 24,
      GLX_STENCIL_SIZE    , 8,
      GLX_DOUBLEBUFFER    , True,
      //GLX_SAMPLE_BUFFERS  , 1,
      //GLX_SAMPLES         , 4,
      None
    };

  int glx_major, glx_minor;

  // FBConfigs were added in GLX version 1.3.
  if ( !glXQueryVersion( display, &glx_major, &glx_minor ) ||
       ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
  {
    printf("Invalid GLX version");
    return 0;
  }

  printf( "Getting matching framebuffer configs\n" );
  int fbcount;
  GLXFBConfig* fbc = glXChooseFBConfig(display, DefaultScreen(display), visual_attribs, &fbcount);
  if (!fbc)
  {
    printf( "Failed to retrieve a framebuffer config\n" );
    return 0;
  }
  printf( "Found %d matching FB configs.\n", fbcount );

  // Pick the FB config/visual with the most samples per pixel
  printf( "Getting XVisualInfos\n" );
  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;

  int i;
  for (i=0; i<fbcount; ++i)
  {
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
    if ( vi )
    {
      int samp_buf, samples;
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );

      printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d, SAMPLES = %d\n", i, (unsigned int) vi -> visualid, samp_buf, samples );

      if ( best_fbc < 0 || samp_buf && samples > best_num_samp )
        best_fbc = i, best_num_samp = samples;
      if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
        worst_fbc = i, worst_num_samp = samples;
    }
    XFree( vi );
  }

  GLXFBConfig bestFbc = fbc[ best_fbc ];

  // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
  XFree( fbc );

  // Get a visual
  XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
  printf( "Chosen visual ID = 0x%x\n",(unsigned int)  vi->visualid );

  printf( "Creating colormap\n" );
  XSetWindowAttributes swa;
  swa.colormap = cmap = XCreateColormap( display,
                                         RootWindow( display, vi->screen ),
                                         vi->visual, AllocNone );
  swa.background_pixmap = None ;
  swa.border_pixel      = 0;
  //swa.event_mask        = StructureNotifyMask;
  swa.event_mask        =  KeyPressMask    | ExposureMask  | ButtonPressMask | StructureNotifyMask;

  printf( "Creating window\n" );
  win = XCreateWindow( display, RootWindow( display, vi->screen ),
                              0, 0, WIDTH /*Width*/, HEIGHT/*Height*/, 0, vi->depth, InputOutput,
                              vi->visual,
                              CWBorderPixel|CWColormap|CWEventMask, &swa );
  if ( !win )
  {
    printf( "Failed to create window.\n" );
    return 0;
  }

  whatToSwap = win;

  // Done with the visual info data
  XFree( vi );

  XStoreName( display, win, "OpenGL3.x+ Control Window" );

  printf( "Mapping window\n" );
  XMapWindow( display, win );

  // Get the default screen's GLX extension list
  const char *glxExts = glXQueryExtensionsString( display,
                                                  DefaultScreen( display ) );

  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
           glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );


  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  ctxErrorOccurred = 0;
  int (*oldHandler)(Display*, XErrorEvent*) =
      XSetErrorHandler(&ctxErrorHandler);

  // Check for the GLX_ARB_create_context extension string and the function.
  // If either is not present, use GLX 1.3 context creation method.
  if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) ||
       !glXCreateContextAttribsARB )
  {
    printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
    ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
  }
  // If it does, try to get a GL 3.0 context!
  else
  {
    int context_attribs[] =
      {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        None
      };

    printf( "Creating context\n" );
    ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                      True, context_attribs );

    // Sync to ensure any errors generated are processed.
    XSync( display, False );
    if ( !ctxErrorOccurred && ctx )
      printf( "Created GL 3.0 context\n" );
    else
    {
      // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
      // When a context version below 3.0 is requested, implementations will
      // return the newest context version compatible with OpenGL versions less
      // than version 3.0.
      // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
      context_attribs[1] = 1;
      // GLX_CONTEXT_MINOR_VERSION_ARB = 0
      context_attribs[3] = 0;

      ctxErrorOccurred = 0;

      printf( "Failed to create GL 3.0 context"
              " ... using old-style GLX context\n" );
      ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                        True, context_attribs );
    }
  }

  // Sync to ensure any errors generated are processed.
  XSync( display, False );

  // Restore the original error handler
  XSetErrorHandler( oldHandler );

  if ( ctxErrorOccurred || !ctx )
  {
    printf( "Failed to create an OpenGL context\n" );
    return 0;
  }

  // Verifying that context is a direct context
  if ( ! glXIsDirect ( display, ctx ) )
  {
    printf( "Indirect GLX rendering context obtained\n" );
  }
  else
  {
    printf( "Direct GLX rendering context obtained\n" );
  }

  printf( "Making context current\n" );
  glXMakeCurrent( display, win, ctx );




 // glClearColor( 0, 0.0, 0, 1 );
 // glClear( GL_COLOR_BUFFER_BIT );
 // glXSwapBuffers ( display, win );

  printf( "GLX3.0 context ready..\n" );
  return 1;
}






int start_glx3_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc,const char **argv)
{
  if (viewWindow==0)
  {
   fprintf(stderr,"start_glx3_stuff with no window..\n");
   display = XOpenDisplay(NULL);
   if (!display)
   {
    printf("Failed to open X display\n");
    return 0;
   }


  int dummy;
  if(!glXQueryExtension(display, &dummy, &dummy))
    {
      fatalErrorGLX3("X server has no OpenGL GLX extension");
      return 0;
    }


  fprintf(stderr,"Will try not to display a window..\n");
  int w=WIDTH, h=HEIGHT;



  int pbuff_visualAttribs[] = {
					              GLX_RENDER_TYPE, GLX_RGBA_BIT,
					              GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                                  GLX_MAX_PBUFFER_WIDTH, w,
                                  GLX_MAX_PBUFFER_HEIGHT, h,
					              GLX_DOUBLEBUFFER, True,
					              GLX_X_RENDERABLE, GL_TRUE,
					              GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
					              GLX_RED_SIZE, 8,
					              GLX_GREEN_SIZE, 8,
					              GLX_BLUE_SIZE, 8,
					              GLX_ALPHA_SIZE, 8,
                                  GLX_DEPTH_SIZE, 24,
					              None
                                 };
/*
      int visualAttribs[]={
                           GLX_RENDER_TYPE, GLX_RGBA_BIT,
                           GLX_MAX_PBUFFER_WIDTH, w,
                           GLX_MAX_PBUFFER_HEIGHT, h,
                           GLX_RED_SIZE, 4,
                           GLX_GREEN_SIZE, 4,
                           GLX_BLUE_SIZE, 4,
                           GLX_DRAWABLE_TYPE,GLX_PBUFFER_BIT,
                           GLX_DEPTH_SIZE, 24,
                           GLX_DOUBLEBUFFER    , True,
                           None
                           };*/

 int numberOfFramebufferConfigurations = 0;
 fprintf(stderr,"glXChooseFBConfig\n");
 GLXFBConfig* fbConfigs = glXChooseFBConfig( display, DefaultScreen(display), pbuff_visualAttribs, &numberOfFramebufferConfigurations );
 if ( (fbConfigs == NULL) || (numberOfFramebufferConfigurations <= 0) )
        {
            fatalErrorGLX3("P-Buffers not supported.\n");
            return 0;
        }

  printf( "Found %d matching FB configs.\n", numberOfFramebufferConfigurations );
  //XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbConfigs[0] );
  //printf( "Chosen visual ID = 0x%x\n", vi->visualid );

  int pbufferAttribs[]={
                           GLX_PBUFFER_WIDTH, w,
                           GLX_PBUFFER_HEIGHT, h,
                           GLX_PRESERVED_CONTENTS, True,
                           GLX_NONE
                          };

 fprintf(stderr,"glXCreatePbuffer\n");
 GLXPbuffer pbuffer = glXCreatePbuffer( display,fbConfigs[0], pbufferAttribs );
 if (pbuffer==0)
        {
          fatalErrorGLX3("glXCreatePbuffer failed..\n");
          return 0;
        }



  // Get the default screen's GLX extension list
  const char *glxExts = glXQueryExtensionsString( display, DefaultScreen( display ) );

  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );


  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  ctxErrorOccurred = 0;
  int (*oldHandler)(Display*, XErrorEvent*) = XSetErrorHandler(&ctxErrorHandler);


  if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) || !glXCreateContextAttribsARB )
  {
    printf( "glXCreateContextAttribsARB() not found ... using old-style GLX context\n" );
    ctx = glXCreateNewContext(
                              display,
                              fbConfigs[0],
                              GLX_RGBA_TYPE,
                              NULL,
                              GL_TRUE
                             );
  }
    else
  // If it does, try to get a GL 3.0 context!
  {
    int context_attribs[] =
      {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        None
      };

    printf( "Creating context\n" );
    ctx = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs );

    // Sync to ensure any errors generated are processed.
    XSync( display, False );
    if ( !ctxErrorOccurred && ctx )
    {
     printf( "Created GL 3.0 context\n" );
    }
    else
    {
      // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
      // When a context version below 3.0 is requested, implementations will
      // return the newest context version compatible with OpenGL versions less
      // than version 3.0.
      // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
      context_attribs[1] = 1;
      // GLX_CONTEXT_MINOR_VERSION_ARB = 0
      context_attribs[3] = 0;

      ctxErrorOccurred = 0;

      printf( "Failed to create GL 3.0 context using old-style GLX context\n" );
      ctx = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs );
    }
  }

  // clean up:
  XFree( fbConfigs );
  // Sync to ensure any errors generated are processed.
  XSync( display, False );

  // Restore the original error handler
  XSetErrorHandler( oldHandler );

  if ( ctxErrorOccurred || !ctx ) { printf( "Failed to create an OpenGL context\n" ); return 0; }

  // Verifying that context is a direct context
  if ( ! glXIsDirect ( display, ctx ) ) { printf( "Indirect GLX rendering context obtained\n" ); } else
                                        { printf( "Direct GLX rendering context obtained\n" );   }

  printf( "Making context current\n" );

  whatToSwap=pbuffer;
  if ( !glXMakeContextCurrent( display, pbuffer, pbuffer, ctx ) )
      {
        fatalErrorGLX3(RED "glXMakeContextCurrent: Could not start rendering to pbuffer fbo" NORMAL);
        return 0;
      }

   if (!glXGetCurrentDrawable())
   {
    fatalErrorGLX3(RED "No drawable selected (pbuffer fbo)\n" NORMAL);

   }

   glClearColor( 0, 0.0, 0, 1 );
   glClear( GL_COLOR_BUFFER_BIT );
   glXSwapBuffers ( display, whatToSwap );

  printf( "GLX3.0 windowless context ready..\n" );
  return 1;
  }


  return start_glx3_stuffWindowed(WIDTH,HEIGHT,argc,argv);
}










int stop_glx3_stuff()
{
  glXMakeCurrent( display, 0, 0 );
  glXDestroyContext( display, ctx );

  XDestroyWindow( display, win );
  XFreeColormap( display, cmap );
  XCloseDisplay( display );
  return 1;
}







int glx3_endRedraw()
{
  if (doubleBufferGLX3) glXSwapBuffers(display, whatToSwap);/* buffer swap does implicit glFlush */
  else glFlush();  /* explicit flush for single buffered case */
  return 1;
}


int glx3_checkEvents()
{
  //GLboolean            needRedraw = GL_FALSE, recalcModelView = GL_TRUE;
  XEvent  event;
     while(XPending(display))
     {
      XNextEvent(display, &event);
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
             //#warning "redraws are not handled ?"
             // needRedraw=GL_TRUE;
          break;
      }
    }; /* loop to compress events */


         return 1;

     return 0;
}






