#include "downloadFromRenderer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>
#include <GL/glu.h>

#include "../Tools/tools.h"
#include "../Tools/save_to_file.h"

#define OPTIMIZE_DEPTH_EXTRACTION 1
#define FLIP_OPEN_GL_IMAGES 1
#define REAL_DEPTH 1

#if REAL_DEPTH
#else
 #warning "getOpenGLDepth call will not return the real Depth but just the depth buffer , please #define REAL_DEPTH 1"
#endif // REAL_DEPTH

#define X_OFFSET 0 //This should always be 0 and probably removed also  :P


//This is a nice blog post on the subject
//http://lektiondestages.blogspot.com/2013/01/reading-opengl-backbuffer-to-system.html



int downloadOpenGLColor(unsigned char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
   if (color==0) { return 0; }  
   if (width==0) { return 0; }  
   if (height==0) { return 0; }  
   
  GLint ext_format, ext_type;
  
  //fprintf(stderr,"glGetIntegerv..\n");
  #warning "GL_IMPLEMENTATION_COLOR_READ_TYPE manually declared .."
  #define GL_IMPLEMENTATION_COLOR_READ_TYPE   		0x8B9A
  #define GL_IMPLEMENTATION_COLOR_READ_FORMAT 		0x8B9B
  glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
  glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);

  //fprintf(stderr,"glReadPixels..\n");
    #if FLIP_OPEN_GL_IMAGES
       char * inverter = (char *) malloc(3*(width-x)*(height-y)*sizeof(char));
       if (inverter==0) { fprintf(stderr,"Could not allocate a buffer to read inverted color\n"); return 0; }

       glReadPixels(x + X_OFFSET, y, width, height, GL_RGB, GL_UNSIGNED_BYTE,inverter);
       //checkFrameGettersForError("Flipped Color Getter");

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


  //fprintf(stderr,"done..\n");
   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"OpenGL error after getOpenGLColor() \n"); }

   return 1;
}

//#warning "TODO : add Horizontal flipping  <- is the output mirrored ?"

int downloadOpenGLZBuffer(unsigned short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height,float depthScale)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    glReadPixels(x + X_OFFSET , y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    //checkFrameGettersForError("Z-Buffer Getter");
    /*
       Not sure I am calculating the correct depth here..
    */
    memset(depth,0 , (width-x)*(height-y)*2 );

    #if FLIP_OPEN_GL_IMAGES
     unsigned int yp = 0;
     unsigned int i=0;
     unsigned int stride = (width-x)*1;
     float tmpF;

     for (yp=0; yp<height; yp++)
       {
         for ( i =0 ; i < (width-x); i ++ )
            {
              //float tmpF=zbuffer[(height-1-yp)*stride+i];
              //tmpF  = (1.0f - zbuffer[(height-1-yp)*stride+i]) * 65534.0;
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




int downloadOpenGLDepth(unsigned short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height,float depthScale)
{
    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0

   if (checkOpenGLError(__FILE__, __LINE__))
      { fprintf(stderr,"getOpenGLDepth() : Error getting depth bias/scale \n"); }

    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    memset(zbuffer,0,(width-x)*(height-y)*sizeof(float));
    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);
    //checkFrameGettersForError("Depth Getter");

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


    float scaleDepthTo = depthScale;// sceneGetDepthScalingPrameter();
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

                  #if SCALE_REAL_DEPTH_OUTPUT
                    depth[(height-yp-1)*width+xp]*=depthMemoryOutputScale;
                  #endif // SCALE_REAL_DEPTH_OUTPUT
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











int downloadOpenGLDepthFromTexture(const char * filename , unsigned int tex , unsigned int width , unsigned int height)
{
   float * pixelsF = (float * ) malloc(sizeof(float) * width * height);

   if (pixelsF!=0)
   {
    glBindTexture(GL_TEXTURE_2D, tex);
    glGetTexImage ( GL_TEXTURE_2D,
                   tex,
                   GL_DEPTH_COMPONENT,
                   GL_FLOAT,
                   pixelsF);
    checkOpenGLError(__FILE__, __LINE__);
    glBindTexture(GL_TEXTURE_2D, 0);

    char * pixelsC = (char * ) malloc(sizeof(char) * 3 * width * height);
    if (pixelsC!=0)
    {
     float * pixelsFPTR = pixelsF;
     char  * pixelPTR = pixelsC;
     char  * pixelLimit = pixelsC+(3 * width * height);

     while (pixelPTR<pixelLimit)
     {
       *pixelPTR  = (unsigned char) *pixelsFPTR;  pixelPTR++;
       *pixelPTR  = (unsigned char) *pixelsFPTR;  pixelPTR++;
       *pixelPTR  = (unsigned char) *pixelsFPTR;  pixelPTR++;
       pixelsFPTR++;
     }


    saveRawImageToFileOGLR(
                           filename,
                           pixelsC ,
                           width,
                           height,
                           1,
                           16
                          );

     free(pixelsC);
      return 1;

    }
   free(pixelsF);
  }
 return 0;
}


int downloadOpenGLColorFromTexture(const char * filename , unsigned int tex, unsigned int width , unsigned int height)
{
   char * pixels3C = (char* ) malloc(sizeof(char) * 4 * width * height);

   if (pixels3C!=0)
   {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, tex);
    glGetTexImage ( GL_TEXTURE_2D,
                   tex,
                   GL_RGB,
                   GL_UNSIGNED_BYTE,
                   pixels3C);
    checkOpenGLError(__FILE__, __LINE__);
    glBindTexture(GL_TEXTURE_2D, 0);

    char * pixelsC = (char * ) malloc(sizeof(char) * 3 * width * height);
    if (pixelsC!=0)
    {
     float * pixels3CPTR = (float*) pixels3C;
     char  * pixelPTR = pixelsC;
     char  * pixelLimit = pixelsC+(3 * width * height);

     while (pixelPTR<pixelLimit)
     {
       *pixelPTR  = (unsigned char) *pixels3CPTR;  pixelPTR++;       pixels3CPTR++;
       *pixelPTR  = (unsigned char) *pixels3CPTR;  pixelPTR++;       pixels3CPTR++;
       *pixelPTR  = (unsigned char) *pixels3CPTR;  pixelPTR++;       pixels3CPTR++;
     }


     saveRawImageToFileOGLR(
                           filename,
                           pixelsC ,
                           width,
                           height,
                           3,
                           8
                          );
      free(pixelsC);

      return 1;

    }
   free(pixels3C);
  }
 return 0;
}













