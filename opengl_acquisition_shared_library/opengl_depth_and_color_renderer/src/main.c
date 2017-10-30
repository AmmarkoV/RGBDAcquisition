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
#include "ModelLoader/model_loader_tri.h"
#include "ModelLoader/model_converter.h"
#include "scene.h"
#include "tools.h"

#include "save_to_file.h"
#include "Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.h"
#include "../../../tools/AmMatrix/matrixCalculations.h"

#include "../../../tools/ImageOperations/imageOps.h"

#include "tiledRenderer.h"

#include "OGLRendererSandbox.h"

#define OPTIMIZE_DEPTH_EXTRACTION 1
#define FLIP_OPEN_GL_IMAGES 1
#define REAL_DEPTH 1

#if REAL_DEPTH
#else
 #warning "getOpenGLDepth call will not return the real Depth but just the depth buffer , please #define REAL_DEPTH 1"
#endif // REAL_DEPTH

#define X_OFFSET 0 //This should always be 0 and probably removed also  :P


#define NORMAL   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

unsigned int openGLGetComplaintsLeft=10;

#define SCALE_REAL_DEPTH_OUTPUT 1
unsigned int snapsPerformed=0;

void internalTest()
{
  testMatrices();
}


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
     float tmpF;

     for (yp=0; yp<height; yp++)
       {
         for ( i =0 ; i < (width-x); i ++ )
            {
              //float tmpF=zbuffer[(height-1-yp)*stride+i];
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
    if (zbuffer==0) { fprintf(stderr,"Could not allocate a zbuffer to read depth\n"); return 0; }
    memset(zbuffer,0,(width-x)*(height-y)*sizeof(float));
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

  #warning "GL_IMPLEMENTATION_COLOR_READ_TYPE manually declared .."
  #define GL_IMPLEMENTATION_COLOR_READ_TYPE   		0x8B9A
  #define GL_IMPLEMENTATION_COLOR_READ_FORMAT 		0x8B9B
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
    saveRawImageToFileOGLR(colorfile,rgb,(width-x),(height-y),3,8);

    if (rgb!=0) { free(rgb); rgb=0; }
    return ;
}



void writeOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
    short * zshortbuffer = (short *) malloc((width-x)*(height-y)*sizeof(short));
    if (zshortbuffer==0) { fprintf(stderr,"Could not allocate a buffer to write depth to file %s \n",depthfile); return; }

    getOpenGLDepth(zshortbuffer,x,y,width,height);

    saveRawImageToFileOGLR(depthfile,zshortbuffer,(width-x),(height-y),1,16);

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

  updateProjectionMatrix();
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



int passUserCommand(const char * command,const char * value)
{
 return 0;
// return handleUserInput(key,state,x,y);
}



int passUserInput(char key,int state,unsigned int x, unsigned int y)
{
 return handleUserInput(key,state,x,y);
}


int setKeyboardControl(int val)
{
  sceneSwitchKeyboardControl(val);
  return 1;
}

int startOGLRendererSandbox(unsigned int width,unsigned int height , unsigned int viewWindow ,const char * sceneFile)
{
  fprintf(stderr,"startOGLRendererSandbox(%u,%u,%u,%s)\n",width,height,viewWindow,sceneFile);
  snapsPerformed=0;

  char * testP=0;
  fprintf(stderr,"starting glx code .. ");
  start_glx_stuff(width,height,viewWindow,0,testP);
  fprintf(stderr,GREEN" ok\n" NORMAL);
  WIDTH=width;
  HEIGHT=height;

  #if FLIP_OPEN_GL_IMAGES
    fprintf(stderr,"This version of OGLRendererSandbox is compiled to flip OpenGL frames to their correct orientation\n");
  #endif

  initializeHardcodedCallLists();

  char defaultSceneFile[] = "Scenes/simple.conf";
  //( char *)   malloc(sizeof(32)*sizeof(char));
  //strncpy(defaultSceneFile,"scene.conf",32);

  if (sceneFile == 0 ) { return initScene(defaultSceneFile);  } else
                       { return initScene(sceneFile);    }

  return 1;
}

int changeOGLRendererGrabMode(unsigned int sequentialModeOn)
{
  sceneIgnoreTime(sequentialModeOn);
}



int seekRelativeOGLRendererSandbox(int devID,signed int seekFrame)
{
  fprintf(stderr,"seekRelativeOGLRendererSandbox(%u,%d) is a stub\n",devID,seekFrame);
  return 0;
}


int seekOGLRendererSandbox(int devID,unsigned int seekFrame)
{
  return sceneSeekTime(seekFrame);
}


int snapOGLRendererSandbox()
{
 ++snapsPerformed;
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before starting to snapOGLRendererSandbox ( frame %u ) \n",snapsPerformed); }
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



int saveSnapshotOfObjects()
{
  char * rgb = (char *) malloc((WIDTH)*(HEIGHT)*sizeof(char)*3);
  if (rgb==0) { fprintf(stderr,"Could not allocate a buffer to write color \n"); return 0; }
  short * zshortbuffer = (short *) malloc((WIDTH)*(HEIGHT)*sizeof(short));
  if (zshortbuffer==0) { fprintf(stderr,"Could not allocate a buffer to write depth \n"); free(rgb); return; }

  getOpenGLColor(rgb, 0, 0, WIDTH,HEIGHT);
  getOpenGLDepth(zshortbuffer,0,0,WIDTH,HEIGHT);

  unsigned int bboxItemsSize=0;
  unsigned int * bbox2D =  getObject2DBoundingBoxList(&bboxItemsSize);

  unsigned int minX,minY,maxX,maxY,pWidth,pHeight;
  if (bbox2D!=0)
  {
      unsigned int i=0;
      unsigned int maxObjs=bboxItemsSize/4;

      for (i=0; i<maxObjs; i++)
      {
        fprintf(stderr,"Dumping snapshot of object %u/%u \n",i,maxObjs);
        minX = bbox2D[0+i*4];
        minY = bbox2D[1+i*4];
        maxX = bbox2D[2+i*4];
        maxY = bbox2D[3+i*4];
        pWidth = abs(maxX-minX);
        pHeight = abs(maxY-minY);


        if (
             (pWidth<WIDTH) &&
             (pHeight<HEIGHT)
           )
        {
         saveTileRGBToFile(0,i,rgb, minX,minY,WIDTH,HEIGHT,pWidth,pHeight);
         saveTileDepthToFile(0,i,zshortbuffer,  minX,minY,WIDTH,HEIGHT,pWidth,pHeight);
        }
      }

      free(bbox2D);
  }

 if (rgb!=0) { free(rgb); rgb=0; }
 if (zshortbuffer!=0) { free(zshortbuffer); zshortbuffer=0; }

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


/*
   --------------------------------------------------------------------------------------
                                    TRAJECTORY GROUND TRUTH COMPARISON SPECIFIC
   --------------------------------------------------------------------------------------
*/

int compareTrajectoryFiles(const char * outputFile , const char * filenameA , const char * filenameB,unsigned int posesToCompare , unsigned totalDistancePerFrame,unsigned int useAngleObjects)
{
  struct VirtualStream * sceneA = createVirtualStream(filenameA,0);
  struct VirtualStream * sceneB = createVirtualStream(filenameB,0);


  if ( (sceneA==0) || (sceneB==0) )
  {
    fprintf(stderr,"Could not read scenes ..!\n");
  } else
  if (sceneA->numberOfObjects != sceneB->numberOfObjects)
  {
    fprintf(stderr,"Inconsistent scenes , with different number of objects..!\n");
  } else
  {


  if (useAngleObjects)
  {
   generateAngleObjectsForVirtualStream(sceneA,"objSphere");
   generateAngleObjectsForVirtualStream(sceneB,"objSphere");
  }


  unsigned int i=0;
 // unsigned int ticks;
  float posStackA[7]={0};
  float posStackB[7]={0};
  float scaleX=1.0,scaleY=1.0,scaleZ=1.0;
  //float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;



  unsigned int timestampToUse=0;



  //Object 0 is camera , so we draw object 1 To numberOfObjects-1

 fprintf(stderr,"Comparing %u objects across %u poses (joints are disregarded) ..!\n",sceneA->numberOfObjects , posesToCompare);
for (timestampToUse=0; timestampToUse<posesToCompare; timestampToUse++)
   {
     float totalDistance=0;
     //i=0 is the camera we dont compare it..!
     for (i=1; i<sceneA->numberOfObjects; i++)
      {
       //struct Model * mod = models[sceneA->object[i].type];
       float * posA = (float*) &posStackA;
       float * posB = (float*) &posStackB;
       if (
             ( calculateVirtualStreamPos(sceneA,i,timestampToUse,posA,0,&scaleX,&scaleY,&scaleZ) ) &&
             ( calculateVirtualStreamPos(sceneB,i,timestampToUse,posB,0,&scaleX,&scaleY,&scaleZ) )
          )
       {
         float distance = calculateDistance(posA[0]*100,posA[1]*100,posA[2]*100,
                                            posB[0]*100,posB[1]*100,posB[2]*100);
         totalDistance+=distance;
         if (!totalDistancePerFrame) { fprintf(stdout,"%u %u %0.5f\n",i,timestampToUse,distance); }
       } else
       {
         fprintf(stderr,"Cannot calculate distance for obj %u\n",i);
       }

     }
     if (totalDistancePerFrame) { fprintf(stdout,"SUM %u %0.5f\n",timestampToUse,totalDistance); }
    }
  }











  if (sceneA!=0) { destroyVirtualStream(sceneA); }
  if (sceneB!=0) { destroyVirtualStream(sceneB); }

 return 1;
}




/*
   --------------------------------------------------------------------------------------
                                    CONVERT MODEL FILE
   --------------------------------------------------------------------------------------
*/
int dumpModelFileH(const char * inputfile,const char * outputfile)
{
  struct OBJ_Model * obj = loadObj("Models/",inputfile,0);


  fprintf(stderr,"Writing output %s \n",outputfile);
    FILE *fd=0;
    fd = fopen(outputfile,"w");
    if (fd!=0)
    {


      fprintf(stderr,"Writing %u vertices .. \n",obj->numVertices);
      fprintf(fd,"const float %sVertices[] = { ",outputfile);
       unsigned int i=0,j=0;
       for(i=0; i<obj->numGroups; i++)
		{
       for(j=0; j<obj->groups[i].numFaces; j++)
			{

				fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].x,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].y,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[0]].z
                       );
				fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].x,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].y,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[1]].z
                       );
				fprintf(fd," %0.4f , %0.4f , %0.4f ",
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].x,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].y,
                            obj->vertexList[ obj->faceList[ obj->groups[i].faceList[j]].v[2]].z
                       );

			  if (j+1<obj->groups[i].numFaces) { fprintf(fd," , \n"); }
			}
		}
        fprintf(fd,"}; \n\n");



        fprintf(stderr,"Writing %u normals .. \n",obj->numNormals);
        fprintf(fd,"const float %sNormals[] = { ",outputfile);

       for(i=0; i<obj->numGroups; i++)
		{
       for(j=0; j<obj->groups[i].numFaces; j++)
			{
              if( obj->groups[i].hasNormals)
                  {
                     fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n1 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n2 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[0]].n3
                              );
                     fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n1 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n2 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[1]].n3
                              );

                     fprintf(fd," %0.4f , %0.4f , %0.4f ",
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n1 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n2 ,
                                 obj->normalList[ obj->faceList[ obj->groups[i].faceList[j]].n[2]].n3
                              );
                  }
				else
				 {
					fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3
                              );
					fprintf(fd," %0.4f , %0.4f , %0.4f , \n",
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3
                              );
					fprintf(fd," %0.4f , %0.4f , %0.4f ",
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n1,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n2,
                                 obj->faceList[ obj->groups[i].faceList[j]].fc_normal.n3
                              );
				}

			  if (j+1<obj->groups[i].numFaces) { fprintf(fd," , \n"); }
			}//FOR J
		}

        fprintf(fd,"}; \n\n");





       fflush(fd);
       fclose(fd);
    }

  unloadObj(obj);
}


int dumpModelFile(const char * inputfile,const char * outputfile)
{
  struct OBJ_Model * obj = loadObj("Models/",inputfile,0);

  if (obj==0)
  {
    fprintf(stderr,"Could not load %s \n",inputfile);
    return 0;
  }
  struct TRI_Model tri={0};

  convertObjToTri(&tri , obj);

  char headerOut[256];
  //

  if (strstr(outputfile,".obj"))
  {
    saveOBJ(obj,outputfile);
  } else
  if (strstr(outputfile,".tri"))
  {
   snprintf(headerOut,256,"%s.tri",outputfile);
   saveModelTri(headerOut,&tri);

   struct TRI_Model triReload={0};
   loadModelTri(headerOut, &triReload);
  } else
  if (strstr(outputfile,".h"))
  {
    dumpModelFileH(inputfile,outputfile);
  }

  unloadObj(obj);
}



