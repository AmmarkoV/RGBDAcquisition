/* A simple program to show how to set up an X window for OpenGL rendering.
 * X86 compilation: gcc -o -L/usr/X11/lib   main main.c shader_loader.c -lGL -lX11
 * X64 compilation: gcc -o -L/usr/X11/lib64 main main.c shader_loader.c -lGL -lX11
 */
#include <stdio.h>
#include <stdlib.h>

#include <GL/glx.h>    /* this includes the necessary X headers */
#include <GL/gl.h>

#include <X11/X.h>    /* X11 constant (e.g. TrueColor) */
#include <X11/keysym.h>

#include "model_loader_obj.h"
#include "scene.h"

#include "shader_loader.h"


#include "OGLRendererSandbox.h"


unsigned int simplePow(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}

int saveRawImageToFile(char * filename,void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (bitsperpixel>16) fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n");
    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            return 1;
        }

        fprintf(fd, "%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
        fwrite(pixels, 1 , n , fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }
    return 0;
}


void WriteOpenGLDepthShort(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{

    short * zbuffer = (short *) malloc((width-x)*(height-y)*sizeof(short));

    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT,zbuffer);
    saveRawImageToFile(depthfile,zbuffer,(width-x),(height-y),1,16);

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }

    return ;
}

int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{

    double depth_bias=0.0; double depth_scale=1.0;
    glGetDoublev(GL_DEPTH_BIAS,  &depth_bias);  // Returns 0.0
    glGetDoublev(GL_DEPTH_SCALE, &depth_scale); // Returns 1.0


    float * zbuffer = (float *) malloc((width-x)*(height-y)*sizeof(float));
    glReadPixels(x, y, width, height, GL_DEPTH_COMPONENT, GL_FLOAT,zbuffer);




    float multiplier = 65536 / (farPlane-nearPlane);

    int i=0;
    for ( i =0 ; i < (width-x)*(height-y); i ++ )
      {
        if (zbuffer[i]>=farPlane-nearPlane)  { depth[i]= 0.0f; } else
                                             { depth[i]=  65536 - zbuffer[i] * multiplier; }
      }

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }
}

void WriteOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{

    short * zshortbuffer = (short *) malloc((width-x)*(height-y)*sizeof(short));

    getOpenGLDepth(zshortbuffer,x,y,width,height);

    saveRawImageToFile(depthfile,zshortbuffer,(width-x),(height-y),1,16);

    if (zshortbuffer!=0) { free(zshortbuffer); zshortbuffer=0; }

    return ;
}



int getOpenGLColor(char * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
     glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE,depth);
}

void redraw(void)
{
    glEnable (GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderScene();

    glx_endRedraw();
}

int entry(int argc, char **argv)
{
  startOGLRendererSandbox();

  /*** (9) dispatch X events ***/
  while (1)
  {
    if (glx_checkEvents())
    {
      tickScene();
      redraw();
    }
  }

  closeScene();

  //unloadShader(depthShaders);

  return 0;
}

double getOpenGLFocalLength()
{
 return nearPlane;
}

double getOpenGLPixelSize()
{
 return 2/WIDTH;
}


int startOGLRendererSandbox()
{
  start_glx_stuff(WIDTH,HEIGHT,0,"");

  initScene();
  return 1;
}


int snapOGLRendererSandbox()
{
    if (glx_checkEvents())
    {
      tickScene();
      redraw();
    }
}

int stopOGLRendererSandbox()
{
  closeScene();
  return 1;
}
