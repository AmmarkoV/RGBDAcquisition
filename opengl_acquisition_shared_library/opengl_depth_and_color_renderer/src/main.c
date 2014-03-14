/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c shader_loader.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c shader_loader.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include "glx.h"
#include "model_loader_obj.h"
#include "scene.h"

#include "save_to_file.h"
#include "shader_loader.h"
#include "AmMatrix/matrixCalculations.h"

#include "tiledRenderer.h"

#include "OGLRendererSandbox.h"

#define FLIP_OPEN_GL_IMAGES 1

#define X_OFFSET 0 //This should always be 0 and probably removed also  :P

//TODO : add Horizontal flipping  <- is the output mirrored ?


int doTest()
{
    testMatrices();
    return ;
}


int getOpenGLZBuffer(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    glReadPixels(x + X_OFFSET , y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    /*
       Not sure I am calculating the correct depth here..
    */
    float max_distance = farPlane-nearPlane;
    float multiplier = (float) 65535 / max_distance;

    memset(depth,0 , (width-x)*(height-y)*2 );

    #if FLIP_OPEN_GL_IMAGES
     unsigned int   yp = 0;
     unsigned int i=0;
     unsigned int stride = (width-x)*1;

     for (yp=0; yp<height; yp++)
       {
         for ( i =0 ; i < (width-x); i ++ )
            {

               if (zbuffer[(height-1-yp)*stride+i]>=max_distance)  { depth[yp*stride+i]=  (short) 0;  } else
                                                                         {
                                                                           float tmpF=zbuffer[(height-1-yp)*stride+i];

                                                                           tmpF  = (1.0f - zbuffer[(height-1-yp)*stride+i]) /* * depthUnit This scales depth but is bad*/ * multiplier;
                                                                           unsigned short tmp = (unsigned short) tmpF;
                                                                           depth[yp*stride+i]= tmp ;
                                                                         }

            }
       }
    #else
    int i=0;
    for ( i =0 ; i < (width-x)*(height-y); i ++ )
      {
        if (zbuffer[i]>=max_distance)  { depth[i]=  (short) 0; } else
                                       {
                                         float tmpF  = (1.0f - zbuffer[i]) * multiplier;
                                         unsigned short tmp = tmpF;
                                         depth[i]= tmp;
                                       }
      }
    #endif

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }
    return 1;
}




int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    memset(zbuffer,0,(width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    /*
       Not sure I am calculating the correct depth here..
    */
    //float max_distance = farPlane-nearPlane;
    //float multiplier = (float) 65535 / max_distance;

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
                if (tmpF!=1.0)
                {
                 gluUnProject((double) xp , (double) yp, (double) tmpF , modelview, projection, viewport, &posX, &posY, &posZ);
                 depth[(height-yp-1)*width+xp]=(unsigned short) posZ;
                }
            }
       }

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }
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
    #if FLIP_OPEN_GL_IMAGES
       char * inverter = (char *) malloc(3*(width-x)*(height-y)*sizeof(char));
       if (inverter==0) { fprintf(stderr,"Could not allocate a buffer to read inverted color\n"); return 0; }

       glReadPixels(x + X_OFFSET, y, width, height, GL_RGB, GL_UNSIGNED_BYTE,inverter);

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
    #endif

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


int startOGLRendererSandbox(unsigned int width,unsigned int height , unsigned int viewWindow ,char * sceneFile)
{
  fprintf(stderr,"startOGLRendererSandbox(%u,%u,%u,%s)\n",width,height,viewWindow,sceneFile);
  testMatrices();

  char test[12]={0};
  char * testP = test;
  start_glx_stuff(width,height,viewWindow,0,&testP);
  WIDTH=width;
  HEIGHT=height;


  #if FLIP_OPEN_GL_IMAGES
    fprintf(stderr,"This version of OGLRendererSandbox is compiled to flip OpenGL frames to their correct orientation\n");
  #endif


  char * defaultSceneFile = "scene.conf";
  //( char *)   malloc(sizeof(32)*sizeof(char));
  //strncpy(defaultSceneFile,"scene.conf",32);

  if (sceneFile == 0 ) { initScene(defaultSceneFile);  } else
                       { initScene(sceneFile);    }

  //free(defaultSceneFile);
  fprintf(stderr,"startOGLRendererSandbox returning\n");
  return 1;
}


int snapOGLRendererSandbox()
{
    if (glx_checkEvents())
    {
      tickScene();
      redraw();
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

