/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c shader_loader.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c shader_loader.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include "glx.h"
#include "ModelLoader/model_loader_obj.h"
#include "scene.h"
#include "tools.h"

#include "save_to_file.h"
#include "shader_loader.h"
#include "../../../tools/AmMatrix/matrixCalculations.h"

#include "tiledRenderer.h"

#include "OGLRendererSandbox.h"

#define OPTIMIZE_DEPTH_EXTRACTION 1
#define FLIP_OPEN_GL_IMAGES 1
#define REAL_DEPTH 1

#define X_OFFSET 0 //This should always be 0 and probably removed also  :P


#define NORMAL   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

unsigned int openGLGetComplaintsLeft=10;

void checkFrameGettersForError(char * from)
{

  int err=glGetError();
  if (err !=  GL_NO_ERROR /*0*/ )
    {
      if (openGLGetComplaintsLeft>0)
      {
        --openGLGetComplaintsLeft;
        fprintf(stderr,YELLOW "Note: OpenGL stack is complaining about the way %s works , this will appear %u more times \n" NORMAL , from , openGLGetComplaintsLeft);
      } else
      {
        openGLGetComplaintsLeft=0;
      }
    }
}

#warning "TODO : add Horizontal flipping  <- is the output mirrored ?"

int getOpenGLZBuffer(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    glReadPixels(x + X_OFFSET , y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    checkFrameGettersForError("Z-Buffer Getter");
    /*
       Not sure I am calculating the correct depth here..
    */
    memset(depth,0 , (width-x)*(height-y)*2 );

    #if FLIP_OPEN_GL_IMAGES
     unsigned int yp = 0;
     unsigned int i=0;
     unsigned int stride = (width-x)*1;

     for (yp=0; yp<height; yp++)
       {
         for ( i =0 ; i < (width-x); i ++ )
            {
              float tmpF=zbuffer[(height-1-yp)*stride+i];
              tmpF  = (1.0f - zbuffer[(height-1-yp)*stride+i]) * 65534.0;
              unsigned short tmp = (unsigned short) tmpF;
              depth[yp*stride+i]= tmp ;
            }
       }
    #else
    int i=0;
    for ( i =0 ; i < (width-x)*(height-y); i ++ )
      {
           float tmpF  = (1.0f - zbuffer[i]) * 65534.0;
           unsigned short tmp = (unsigned short) tmpF;
           depth[i]= tmp;
      }
    #endif

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }


   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"OpenGL error after getOpenGLZBuffer() \n"); }
    return 1;
}




int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"getOpenGLDepth() : Error getting depth bias/scale \n"); }

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    memset(zbuffer,0,(width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    checkFrameGettersForError("Depth Getter");

    /*
       Not sure I am calculating the correct depth here..
    */

    memset(depth,0 , (width-x)*(height-y)*2 );


    GLint viewport[4];
    GLdouble modelview[16];
    GLdouble projection[16];
    GLdouble posX, posY, posZ=0.0;

    glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
    glGetDoublev( GL_PROJECTION_MATRIX, projection );
    glGetIntegerv( GL_VIEWPORT, viewport );

     unsigned int xp = 0, yp = 0;

     for (yp=0; yp<height; yp++)
       {
         for ( xp=0 ; xp<width; xp++)
            {
                float tmpF=zbuffer[yp*width+xp];

                #if OPTIMIZE_DEPTH_EXTRACTION
                if (tmpF==0)
                {
                  //Do nothing , fast path
                } else
                #endif // OPTIMIZE_DEPTH_EXTRACTION
                if (tmpF<depth_scale)
                {
                 gluUnProject((double) xp , (double) yp, (double) tmpF , modelview, projection, viewport, &posX, &posY, &posZ);

                 #if REAL_DEPTH
                  tmpF = sqrt( (posX*posX) + (posY*posY) + (posZ*posZ) ) * scaleDepthTo;
                  depth[(height-yp-1)*width+xp]=(unsigned short) tmpF;
                 #else
                  depth[(height-yp-1)*width+xp]=(unsigned short) posZ * scaleDepthTo;
                 #endif // REAL_DEPTH

                }
            }
       }

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }


   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"OpenGL error after getOpenGLDepth() \n"); }

    return 1;
}


unsigned int getOpenGLWidth()
{
    return WIDTH;
}

unsigned int getOpenGLHeight()
{
    return HEIGHT;
}

int getOpenGLColor(char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
  GLint ext_format, ext_type;
  glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
  glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);

    #if FLIP_OPEN_GL_IMAGES
       char * inverter = (char *) malloc(3*(width-x)*(height-y)*sizeof(char));
       if (inverter==0) { fprintf(stderr,"Could not allocate a buffer to read inverted color\n"); return 0; }

       glReadPixels(x + X_OFFSET, y, width, height, GL_RGB, GL_UNSIGNED_BYTE,inverter);
       checkFrameGettersForError("Flipped Color Getter");

      //SLOW INVERSION CODE :P
       unsigned int yp = 0;
       unsigned int stride = (width-x)*3;

       for (yp=0; yp<height; yp++)
       {
         char * where_to = &color[yp*stride];
         char * where_from = &inverter[(height-1-yp)*stride];
         memcpy(where_to , where_from , stride * sizeof(char));
       }
      free(inverter);
    #else
       glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE,color);
       checkFrameGettersForError("Normal Color Getter");
    #endif


   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"OpenGL error after getOpenGLColor() \n"); }

   return 1;
}


void writeOpenGLColor(char * colorfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{

    char * rgb = (char *) malloc((width-x)*(height-y)*sizeof(char)*3);
    if (rgb==0) { fprintf(stderr,"Could not allocate a buffer to write color to file %s \n",colorfile); return ; }

    getOpenGLColor(rgb, x, y, width,  height);
    saveRawImageToFile(colorfile,rgb,(width-x),(height-y),3,8);

    if (rgb!=0) { free(rgb); rgb=0; }
    return ;
}



void writeOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    short * zshortbuffer = (short *) malloc((width-x)*(height-y)*sizeof(short));
    if (zshortbuffer==0) { fprintf(stderr,"Could not allocate a buffer to write depth to file %s \n",depthfile); return; }

    getOpenGLDepth(zshortbuffer,x,y,width,height);

    saveRawImageToFile(depthfile,zshortbuffer,(width-x),(height-y),1,16);

    if (zshortbuffer!=0) { free(zshortbuffer); zshortbuffer=0; }

    return ;
}




void redraw(void)
{
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error just after receiving a redraw command\n"); }
    renderScene();
    glx_endRedraw();
}


int setOpenGLNearFarPlanes(double near , double far)
{
 farPlane=far;
 nearPlane=near;
 return 1;
}

int setOpenGLIntrinsicCalibration(double * camera)
{
  useIntrinsicMatrix=1;
  cameraMatrix[0]=camera[0];
  cameraMatrix[1]=camera[1];
  cameraMatrix[2]=camera[2];
  cameraMatrix[3]=camera[3];
  cameraMatrix[4]=camera[4];
  cameraMatrix[5]=camera[5];
  cameraMatrix[6]=camera[6];
  cameraMatrix[7]=camera[7];
  cameraMatrix[8]=camera[8];
  return 1;
}


int setOpenGLExtrinsicCalibration(double * rodriguez,double * translation , double scaleToDepthUnit)
{
  useCustomModelViewMatrix=1;
  convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(customModelViewMatrix , rodriguez , translation , scaleToDepthUnit);

  scaleDepthTo = (float) scaleToDepthUnit;

  customTranslation[0] = translation[0];
  customTranslation[1] = translation[1];
  customTranslation[2] = translation[2];

  customRodriguezRotation[0] = rodriguez[0];
  customRodriguezRotation[1] = rodriguez[1];
  customRodriguezRotation[2] = rodriguez[2];
  return 1;
}


double getOpenGLFocalLength()
{
 return nearPlane;
}

double getOpenGLPixelSize()
{
 return 2/WIDTH;
}



int setKeyboardControl(int val)
{
  sceneSwitchKeyboardControl(val);
  return 1;
}

int enableShaders(char * vertShaderFilename , char * fragShaderFilename)
{
  strncpy(fragmentShaderFile , fragShaderFilename,MAX_FILENAMES);
  selectedFragmentShader = fragmentShaderFile;

  strncpy(vertexShaderFile , vertShaderFilename,MAX_FILENAMES);
  selectedVertexShader = vertexShaderFile;

  return 1;
}

int startOGLRendererSandbox(unsigned int width,unsigned int height , unsigned int viewWindow ,char * sceneFile)
{
  fprintf(stderr,"startOGLRendererSandbox(%u,%u,%u,%s)\n",width,height,viewWindow,sceneFile);

  char test[12]={0};
  char * testP = test;
  start_glx_stuff(width,height,viewWindow,0,&testP);
  WIDTH=width;
  HEIGHT=height;


  #if FLIP_OPEN_GL_IMAGES
    fprintf(stderr,"This version of OGLRendererSandbox is compiled to flip OpenGL frames to their correct orientation\n");
  #endif


  char defaultSceneFile[] = "scene.conf";
  //( char *)   malloc(sizeof(32)*sizeof(char));
  //strncpy(defaultSceneFile,"scene.conf",32);

  if (sceneFile == 0 ) { return initScene(defaultSceneFile);  } else
                       { return initScene(sceneFile);    }

  return 1;
}


int snapOGLRendererSandbox()
{
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before starting to snapOGLRendererSandbox \n"); }
    if (glx_checkEvents())
    {
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after checking glx_checkEvents()\n"); }
      tickScene();
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after ticking scene\n"); }
      redraw();
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after redrawing scene\n"); }
      return 1;
    }
   return 0;
}



int stopOGLRendererSandbox()
{
  closeScene();
  return 1;
}



/*
   --------------------------------------------------------------------------------------
                                    PHOTOSHOOT SPECIFIC
   --------------------------------------------------------------------------------------
*/




void * createOGLRendererPhotoshootSandbox(
                                           int objID, unsigned int columns , unsigned int rows , float distance,
                                           float angleX,float angleY,float angleZ,
                                           float angXVariance ,float angYVariance , float angZVariance
                                         )
{
  return createPhotoshoot(
                           objID,
                           columns , rows ,
                           distance,
                           angleX,angleY,angleZ ,
                           angXVariance ,angYVariance , angZVariance
                         );
}

int destroyOGLRendererPhotoshootSandbox( void * photoConf )
{
    fprintf(stderr,"destroyOGLRendererPhotoshootSandbox is a stub : %p \n",photoConf);
   return 0;
}


int getOGLPhotoshootTileXY(void * photoConf , unsigned int column , unsigned int row ,
                                              float * X , float * Y)
{
  float x2D , y2D , z2D;
  tiledRenderer_get2DCenter(photoConf,column,row,&x2D,&y2D,&z2D);

  //fprintf(stderr,"Column/Row %u/%u -> %0.2f %0.2f %0.2f\n",column,row , x2D , y2D , z2D);
  *X = x2D;
  *Y = y2D;
  return 1;
}



int snapOGLRendererPhotoshootSandbox(
                                     void * photoConf ,
                                     int objID, unsigned int columns , unsigned int rows , float distance,
                                     float angleX,float angleY,float angleZ,
                                     float angXVariance ,float angYVariance , float angZVariance
                                    )
{

    setupPhotoshoot( photoConf , objID, columns , rows , distance,
                     angleX, angleY, angleZ , angXVariance , angYVariance , angZVariance );


    if (glx_checkEvents())
    {
      renderPhotoshoot(photoConf);
      return 1;
    }
   return 0;
}

