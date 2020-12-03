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

#include "System/glx.h"
#include "ModelLoader/model_loader_hardcoded.h"
#include "ModelLoader/model_loader_obj.h"
#include "ModelLoader/model_loader_tri.h"
#include "ModelLoader/model_converter.h"

#include "Scene/scene.h"
#include "Scene/photoShootingScene.h"

#include "Rendering/downloadFromRenderer.h"

#include "Tools/tools.h"
#include "Tools/save_to_file.h"

 
#include "TrajectoryParser/TrajectoryParserDataStructures.h"
#include "TrajectoryParser/TrajectoryCalculator.h"




#include "Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.h"
#include "Rendering/tiledRenderer.h"

#include "../../../../tools/AmMatrix/matrixCalculations.h"

#include "../../../../tools/ImageOperations/imageOps.h"


#include "OGLRendererSandbox.h"


#define NORMAL   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

unsigned int openGLVersion=3;
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

int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
   return downloadOpenGLDepth((unsigned short*)depth,x,y,width,height,sceneGetDepthScalingPrameter());
}

unsigned int getOpenGLWidth()
{
    return WIDTH;
}

unsigned int getOpenGLHeight()
{
    return HEIGHT;
}

unsigned int getOpenGLTimestamp()
{
    return getLoadedScene()->timestampToUse;
}

int getOpenGLColor(char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{
   return downloadOpenGLColor((unsigned char*) color,x,y,width,height);
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
    if (zshortbuffer==0) { fprintf(stderr,"Could not allocate a buffer to write depth to file %s \n",depthfile); return; } else
        {
         getOpenGLDepth(zshortbuffer,x,y,width,height);

         saveRawImageToFileOGLR(depthfile,zshortbuffer,(width-x),(height-y),1,16);

         free(zshortbuffer); zshortbuffer=0;
        }


    return ;
}




void redraw(void)
{
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error just after receiving a redraw command\n"); }
    renderScene();
  glx_endRedraw(openGLVersion);
}


int setOpenGLNearFarPlanes(float near ,float far)
{
 return sceneSetNearFarPlanes((float) near, (float) far);
}

int setOpenGLIntrinsicCalibration(float * camera)
{
  return sceneSetOpenGLIntrinsicCalibration(getLoadedScene(),camera);
}

int setOpenGLIntrinsicCalibrationNew(float fx,float fy,float cx,float cy,float width,float height,float nearPlane,float farPlane)
{
  fprintf(stderr,"setOpenGLIntrinsicCalibrationNew: %0.1fx%0.1f\n",width,height);
  fprintf(stderr,"fx:%0.2f|fy:%0.2f|cx:%0.2f|cy:%0.2f\n",fx,fy,cx,cy);
  return sceneSetOpenGLIntrinsicCalibrationNew(getLoadedScene(),fx,fy,cx,cy,width,height,nearPlane,farPlane);
}



int setOpenGLExtrinsicCalibration(float * rodriguez,float * translation , float scaleToDepthUnit)
{
  return sceneSetOpenGLExtrinsicCalibration(getLoadedScene(),rodriguez,translation,scaleToDepthUnit);
}


float getOpenGLFocalLength()
{
 return sceneGetNearPlane();
}

float getOpenGLPixelSize()
{
 return (float) 2/WIDTH;
}

int controlScene(const char * name,const char * variable,int control,float valueA,float valueB,float valueC)
{
 if (name==0) { return 0; }
 if (variable==0) { return 0; }
 //TODO:
 /* THIS SHOULD BE THE SAME AS OpenGLAcquisition.h
  OPENGL_ACQUISITION_NOCONTROL=0,          //0
  OPENGL_ACQUISITION_POSITION_XYZ,         //1
  OPENGL_ACQUISITION_ROTATION_XYZ,         //2
  OPENGL_ACQUISITION_JOINT_ROTATION_XYZ,   //3
  OPENGL_ACQUISITION_JOINT_ROTATION_ZXY,   //4
  OPENGL_ACQUISITION_JOINT_ROTATION_ZYX,   //5
  OPENGL_ACQUISITION_JOINT_ROTATION_TEST,  //6
  OPENGL_ACQUISITION_COLOR_RGB,            //7
*/
 float coords[4]={0};
 char doRotationControl=0;
  switch(control)
  {
     case 0: //OPENGL_ACQUISITION_NOCONTROL
       return 0;
     break;
     case 1: //OPENGL_ACQUISITION_POSITION_XYZ
       coords[0]=valueA;
       coords[1]=valueB;
       coords[2]=valueC;
       return moveAllPosesInObjectState(getLoadedScene(),getLoadedModelStorage(),name,0,coords,3);
     break;
     case 2: //OPENGL_ACQUISITION_ROTATION_XYZ
       fprintf(stderr,"OPENGL_ACQUISITION_ROTATION_XYZ not implemented..!\n");
       return 0;
     break;
     case 3: //OPENGL_ACQUISITION_JOINT_ROTATION_XYZ
       coords[0]=valueA;
       coords[1]=valueB;
       coords[2]=valueC;
       doRotationControl=1;
     break;
     case 4: //OPENGL_ACQUISITION_JOINT_ROTATION_ZXY
       coords[0]=valueC;
       coords[1]=valueA;
       coords[2]=valueB;
       doRotationControl=1;
     break;
     case 5: //OPENGL_ACQUISITION_JOINT_ROTATION_ZYX
       coords[0]=valueC;
       coords[1]=valueB;
       coords[2]=valueA;
       doRotationControl=1;
     break;
     case 6: //OPENGL_ACQUISITION_JOINT_ROTATION_NEGATIVE_XYZ
       coords[0]=-1*valueA;
       coords[1]=-1*valueB;
       coords[2]=-1*valueC;
       doRotationControl=1;
     break;
     case 7: //OPENGL_ACQUISITION_COLOR_RGB
       fprintf(stderr,"OPENGL_ACQUISITION_COLOR_RGB not implemented..!\n");
     break;
     default :
       fprintf(stderr,"Unhandled control (%u) for controlScene call\n",control);
       return 0;
  };

  if (doRotationControl)
      { return changeAllPosesInObjectState(getLoadedScene(),getLoadedModelStorage(),name,variable,0,coords,3); }
  return 0;
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



int startOGLRendererSandbox(int argc,const char *argv[],unsigned int width,unsigned int height , unsigned int viewWindow ,const char * sceneFile)
{
  //fprintf(stderr,"startOGLRendererSandbox(%u,%u,%u,%s)\n",width,height,viewWindow,sceneFile);
  snapsPerformed=0;

  //char ** testP=0;
  fprintf(stderr,"trying to start glx code with configuration ( %ux%u , viewWindow=%u )..\n",width,height,viewWindow);
   if ( !start_glx_stuff(width,height,openGLVersion,viewWindow,argc,argv/*0,testP*/) )
   {
     return 0;
   }
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

  if (sceneFile == 0 ) { return initScene(argc,argv,defaultSceneFile);  } else
                       { return initScene(argc,argv,sceneFile);    }

  return 1;
}

int changeOGLRendererGrabMode(unsigned int sequentialModeOn)
{
  return  sceneIgnoreTime(sequentialModeOn);
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


int snapOGLRendererSandbox(unsigned int framerate)
{
 ++snapsPerformed;
 if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error before starting to snapOGLRendererSandbox ( frame %u ) \n",snapsPerformed); }
    if (glx_checkEvents(openGLVersion))
    {
      if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"OpenGL error after checking glx_checkEvents()\n"); }
      tickScene(framerate);
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
  if (zshortbuffer==0) { fprintf(stderr,"Could not allocate a buffer to write depth \n"); free(rgb); return 0; }

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
         saveTileRGBToFile(0,i,(unsigned char*) rgb,minX,minY,WIDTH,HEIGHT,pWidth,pHeight);
         saveTileDepthToFile(0,i,(unsigned short*) zshortbuffer,minX,minY,WIDTH,HEIGHT,pWidth,pHeight);
        }
      }

      free(bbox2D);
  }

 if (rgb!=0) { free(rgb); rgb=0; }
 if (zshortbuffer!=0) { free(zshortbuffer); zshortbuffer=0; }
 return 1;
}










/*
   --------------------------------------------------------------------------------------
                                    PHOTOSHOOT SPECIFIC
   --------------------------------------------------------------------------------------
*/




void * createOGLRendererPhotoshootSandbox(
                                           void * scene,
                                           void * modelStorage,
                                           int objID, unsigned int columns , unsigned int rows , float distance,
                                           float angleX,float angleY,float angleZ,
                                           float angXVariance ,float angYVariance , float angZVariance
                                         )
{
  return createPhotoshoot(
                           scene,
                           modelStorage,
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


    if (glx_checkEvents(openGLVersion))
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
    //This is broken ? 3/6/2018
//   generateAngleObjectsForVirtualStream(sceneA,"objSphere");
//   generateAngleObjectsForVirtualStream(sceneB,"objSphere");
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
  int result=0;
  struct OBJ_Model * obj = loadObj("Models/",inputfile,0);


  fprintf(stderr,"Writing output %s \n",outputfile);
    FILE *fd=0;
    fd = fopen(outputfile,"w");
    if (fd!=0)
    {


      fprintf(stderr,"Writing %lu vertices .. \n",obj->numVertices);
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



        fprintf(stderr,"Writing %lu normals .. \n",obj->numNormals);
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
       result=1;
    }

  unloadObj(obj);
  return result;
}


int dumpModelFile(const char * inputfile,const char * outputfile)
{
  int result =0 ;
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
    result = saveOBJ(obj,outputfile);
  } else
  if (strstr(outputfile,".tri"))
  {
   snprintf(headerOut,256,"%s.tri",outputfile);
   result = saveModelTri(headerOut,&tri);

   struct TRI_Model triReload={0};
   loadModelTri(headerOut, &triReload);
  } else
  if (strstr(outputfile,".h"))
  {
    result = dumpModelFileH(inputfile,outputfile);
  }

  unloadObj(obj);
  return result;
}



