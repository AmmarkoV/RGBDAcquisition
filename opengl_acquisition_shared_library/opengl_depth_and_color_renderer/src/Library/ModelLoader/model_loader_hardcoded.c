
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <math.h>

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "model_loader.h"
#include "model_loader_hardcoded.h"
#include "hardcoded_shapes.h"

#include "model_loader_obj.h"
#include "../Tools/tools.h"

#include "../Rendering/ogl_rendering.h"
#include "../../../../../tools/AmMatrix/matrixCalculations.h"

#define DISABLE_GL_CALL_LIST 0
#if DISABLE_GL_CALL_LIST
 #warning "Please note that glCallList is disabled and that has a really bad effect on graphics card performance"
#endif // DISABLE_GL_CALL_LIST

#define SPHERE_QUALITY 10 /*100 is good quality*/

#define PIE 3.14159265358979323846
#define degreeToRadOLD(deg) (deg)*(PIE/180)



#define USE_QUESTIONMARK_FOR_FAILED_LOADED_MODELS 1

GLuint hardcodedObjlist[TOTAL_POSSIBLE_MODEL_TYPES]={0};



#define OLD_WAY_TO_DRAW 0



float sphereCoords[SPHERE_QUALITY*SPHERE_QUALITY*3]={0};
float sphereNormals[SPHERE_QUALITY*SPHERE_QUALITY*3]={0};



int calculateGenericTriangleNormals(const float * coords , unsigned int coordLength)
{
 if (coords==0) { return 0; }
   //fprintf(stderr,"float xNormals[]={ //X  Y  Z  W\n");
    float outputNormal[4]={0};
    unsigned int i=0,z=0,z1=0,z2=0,z3=0;
      for (i=0; i<coordLength/3; i++)
        {
          z=(i*3)*3;  z1=z;  outputNormal[0]=coords[z1+0];  outputNormal[1]=coords[z1+1]; outputNormal[2]=coords[z1+2];  outputNormal[3]=1.0f;
          z+=3;       z2=z;
          z+=3;       z3=z;

          findNormal(&outputNormal[0], &outputNormal[1], &outputNormal[2] ,
                     coords[z1+0]   , coords[z1+1]   , coords[z1+2],
                     coords[z2+0]   , coords[z2+1]   , coords[z2+2],
                     coords[z3+0]   , coords[z3+1]   , coords[z3+2]
                     );

          //fprintf(stderr,"                      %0.4ff,%0.4ff,%0.4ff,\n",outputNormal[0],outputNormal[1],outputNormal[2]);
          //fprintf(stderr,"                      %0.4ff,%0.4ff,%0.4ff,\n",outputNormal[0],outputNormal[1],outputNormal[2]);
          //fprintf(stderr,"                      %0.4ff,%0.4ff,%0.4ff,\n",outputNormal[0],outputNormal[1],outputNormal[2]);
        }
   //fprintf(stderr,"};\n");
  return 1;
}


int drawGenericTriangleMesh(const float * coords ,const float * normals, unsigned int coordLength)
{
    #if USE_GLEW
    return  renderOGL(
                      0,//float * projectionMatrix ,
                      0,//float * viewMatrix ,
                      0,//float * modelMatrix ,
                      0,//float * mvpMatrix ,
                      //-------------------------------------------------------
                      coords ,      coordLength ,
                      normals ,     coordLength ,
                      0 ,         0,
                      0 ,         0,
                      0 ,         0
                     );
    #else
    glBegin(GL_TRIANGLES);
      unsigned int i=0,z=0;
      for (i=0; i<coordLength/3; i++)
        {
                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z=(i*3)*3;  glVertex3f(coords[z+0],coords[z+1],coords[z+2]);

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(coords[z+0],coords[z+1],coords[z+2]);

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(coords[z+0],coords[z+1],coords[z+2]);
        }
    glEnd();
    #endif

    return 1;
}

int drawGenericTriangleMeshTranslatedScaled(const float * coords ,const float * normals, unsigned int coordLength,float dx,float dy,float dz,float scale)
{
    glBegin(GL_TRIANGLES);
      unsigned int i=0,z=0;
      for (i=0; i<coordLength/3; i++)
        {
                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z=(i*3)*3;  glVertex3f(scale*(dx+coords[z+0]),scale*(dy+coords[z+1]),scale*(dz+coords[z+2]));

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(scale*(dx+coords[z+0]),scale*(dy+coords[z+1]),scale*(dz+coords[z+2]));

                      glNormal3f(normals[i+0],normals[i+1],normals[i+2]);
          z+=3;       glVertex3f(scale*(dx+coords[z+0]),scale*(dy+coords[z+1]),scale*(dz+coords[z+2]));
        }
    glEnd();
    return 1;
}


int drawAxis(
             const float x,
             const float y,
             const float z, 
             const float scale
            )
{
 glLineWidth(6.0);
 glBegin(GL_LINES);
  glColor3f(1.0,0.0,0.0); glVertex3f(x,y,z); glVertex3f(x+scale,y,z); //Red is X
  glColor3f(0.0,1.0,0.0); glVertex3f(x,y,z); glVertex3f(x,y+scale,z); //Green is Y
  glColor3f(0.0,0.0,1.0); glVertex3f(x,y,z); glVertex3f(x,y,z+scale); //Blue is Z
 glEnd();
 glLineWidth(1.0);
 return 1;
}

int drawGridPlane(float x,float y,float z , float scale)
{
 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);

  signed int i;

 float floorWidth = 500;
 for(i=-floorWidth; i<=floorWidth; i++)
 {
    glVertex3f(x+-floorWidth*scale,y,z+i*scale);
    glVertex3f(x+floorWidth*scale,y,z+i*scale);

    glVertex3f(x+i*scale,y,z-floorWidth*scale);
    glVertex3f(x+i*scale,y,z+floorWidth*scale);
 };
glEnd();
 return 1;
}

void drawSphere(unsigned int quality)
{
  #if DISABLE_GL_CALL_LIST
    //GL_CALL_LIST is disabled so this is actually ok now
  #else
    #warning "drawSphere should create a call list , otherwise it is a very slow call.."
  #endif

  //float r=1.0;
    int lats=quality;
    int longs=quality;
  //---------------
    int i, j;
    for(i = 0; i <= lats; i++)
    {
       float lat0 = M_PI * (-0.5 + (float) (i - 1) / lats);
       float z0  = sin(lat0);
       float zr0 =  cos(lat0);

       float lat1 = M_PI * (-0.5 + (float) i / lats);
       float z1 = sin(lat1);
       float zr1 = cos(lat1);

       glBegin(GL_QUAD_STRIP);
       for(j = 0; j <= longs; j++)
        {
           float lng = 2 * M_PI * (float) (j - 1) / longs;
           float x = cos(lng);
           float y = sin(lng);

           glNormal3f(x * zr0, y * zr0, z0);
           glVertex3f(x * zr0, y * zr0, z0);
           glNormal3f(x * zr1, y * zr1, z1);
           glVertex3f(x * zr1, y * zr1, z1);
        }
       glEnd();
   }
}



void initializeSphere()
{
  fprintf(stderr,"TODO : initializeSphere not implemented because it has quads instead of triangles..\n");
  return;
//    float r=1.0;
    int lats=SPHERE_QUALITY;
    int longs=SPHERE_QUALITY;
  //---------------
    int i, j;
    for(i = 0; i <= lats; i++)
    {
       float lat0 = M_PI * (-0.5 + (float) (i - 1) / lats);
       float z0  = sin(lat0);
       float zr0 =  cos(lat0);

       float lat1 = M_PI * (-0.5 + (float) i / lats);
       float z1 = sin(lat1);
       float zr1 = cos(lat1);

       glBegin(GL_QUAD_STRIP);
       for(j = 0; j <= longs; j++)
        {
           float lng = 2 * M_PI * (float) (j - 1) / longs;
           float x = cos(lng);
           float y = sin(lng);

           glNormal3f(x * zr0, y * zr0, z0);
           glVertex3f(x * zr0, y * zr0, z0);
           glNormal3f(x * zr1, y * zr1, z1);
           glVertex3f(x * zr1, y * zr1, z1);
        }
       glEnd();
   }
}


int drawQuestion()
{
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , 0 , 0   , 0 ,0.5 );
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , 0 , -3*U , 0 ,0.5 );
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , U/2 , U   , 0 ,0.5);
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , U , 2*U   , 0 ,0.3);
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , U , 2.5*U   , 0 ,0.5);
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , U ,2.8*U   , 0 ,0.3);
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , 0 , 3*U   , 0 ,0.5);
  drawGenericTriangleMeshTranslatedScaled(cubeCoords, cubeNormals,sizeof(cubeCoords)/(3*sizeof(float)) , -U ,2.8*U   , 0 ,0.5);
 return 1;
}

int drawBoundingBox(
                    const float x,
                    const float y,
                    const float z,
                    const float minX,
                    const float minY,
                    const float minZ,
                    const float maxX,
                    const float maxY,
                    const float maxZ
                   )
{
 //fprintf(stderr,"drawBoundingBox( pos %0.2f %0.2f %0.2f min %0.2f %0.2f %0.2f  max %0.2f %0.2f %0.2f \n",x,y,z ,minX,minY,minZ,maxX,maxY,maxZ);

glLineWidth(6.0);

 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+minY,z+minZ);
 glVertex3f(x+minX,y+maxY,z+minZ);
 glVertex3f(x+maxX,y+maxY,z+minZ);
 glVertex3f(x+maxX,y+maxY,z+maxZ);
 glVertex3f(x+minX,y+maxY,z+maxZ);
 glVertex3f(x+minX,y+maxY,z+minZ);
 glEnd();

 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+minY,z+maxZ);
 glVertex3f(x+minX,y+minY,z+minZ);
 glVertex3f(x+maxX,y+minY,z+minZ);
 glVertex3f(x+maxX,y+minY,z+maxZ);
 glVertex3f(x+minX,y+minY,z+maxZ);
 glEnd();


 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+minY,z+maxZ);
 glVertex3f(x+maxX,y+minY,z+maxZ);
 glEnd();

 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+maxY,z+maxZ);
 glVertex3f(x+maxX,y+maxY,z+maxZ);
 glEnd();

 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+minY,z+minZ);
 glVertex3f(x+maxX,y+minY,z+minZ);
 glEnd();

 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+maxY,z+minZ);
 glVertex3f(x+maxX,y+maxY,z+minZ);
 glEnd();



 glBegin(GL_LINES);
 glNormal3f(0.0,1.0,0.0);
 glVertex3f(x+minX,y+minY,z+minZ);
 glVertex3f(x+minX,y+minY,z+maxZ);
 glVertex3f(x+maxX,y+maxY,z+minZ);
 glVertex3f(x+maxX,y+maxY,z+maxZ);
 glVertex3f(x+minX,y+minY,z+minZ);
 glVertex3f(x+minX,y+minY,z+maxZ);
 glVertex3f(x+maxX,y+maxY,z+minZ);
 glVertex3f(x+maxX,y+maxY,z+maxZ);
 glEnd();


glLineWidth(1.0);

return 1;
}


int drawObjPlane(float x,float y,float z,float dimension)
{
    calculateGenericTriangleNormals(planeCoords , sizeof(planeCoords)/(3*sizeof(float)) );

    return drawGenericTriangleMesh(planeCoords , planeNormals , sizeof(planeCoords)/(3*sizeof(float)) );
}

int drawCube()
{
    calculateGenericTriangleNormals(cubeCoords , sizeof(cubeCoords)/(3*sizeof(float)) );

    return drawGenericTriangleMesh(cubeCoords , cubeNormals , sizeof(cubeCoords)/(3*sizeof(float)) );
}

int drawPyramid()
{
    calculateGenericTriangleNormals(pyramidCoords , sizeof(pyramidCoords)/(3*sizeof(float)) );

    return drawGenericTriangleMesh(pyramidCoords , pyramidNormals , sizeof(pyramidCoords)/(3*sizeof(float)) );
}


void drawFog()
{
  glEnable(GL_FOG);
  GLfloat fogColor[4] = {0.5, 0.5, 0.5, 1.0};
  glFogi (GL_FOG_MODE, GL_EXP );
  glFogfv (GL_FOG_COLOR, fogColor);
  glFogf (GL_FOG_DENSITY, 0.35);
  glHint (GL_FOG_HINT, GL_DONT_CARE);
  glFogf (GL_FOG_START, 1.0);
  glFogf (GL_FOG_END, 5.0);
}

unsigned int isModelnameAHardcodedModel(const char * modelname,unsigned int * itIsAHardcodedModel)
{
  *itIsAHardcodedModel=1;
  unsigned int modType=0;
   if ( strcmp(modelname,"fog") == 0 )        {  modType = OBJ_FOG;       }  else
   if ( strcmp(modelname,"plane") == 0 )      {  modType = OBJ_PLANE;     }  else
   if ( strcmp(modelname,"grid") == 0 )       {  modType = OBJ_GRIDPLANE; }  else
   if ( strcmp(modelname,"cube") == 0 )       {  modType = OBJ_CUBE;      }  else
   if ( strcmp(modelname,"pyramid") == 0 )    {  modType = OBJ_PYRAMID;   }  else
   if ( strcmp(modelname,"axis") == 0 )       {  modType = OBJ_AXIS;      }  else
   if ( strcmp(modelname,"sphere") == 0 )     {  modType = OBJ_SPHERE;    }  else
   if ( strcmp(modelname,"none") == 0 )       {  modType = OBJ_INVISIBLE; }  else
   if ( strcmp(modelname,"invisible") == 0 )  {  modType = OBJ_INVISIBLE; }  else
   if ( strcmp(modelname,"question") == 0 )   {  modType = OBJ_QUESTION;  }  else
   if ( strcmp(modelname,"bbox") == 0 )       {  modType = OBJ_BBOX;      }  else
                                              {  *itIsAHardcodedModel=0;   }
  return modType;
}


unsigned int drawHardcodedModelRaw(unsigned int modelType)
{
    switch (modelType)
    {
      case OBJ_FOG       : drawFog();                            break;
      case OBJ_PLANE     : drawObjPlane(0,0,0, 0.5);             break;
      case OBJ_GRIDPLANE : drawGridPlane( 0.0 , 0.0 , 0.0, 1.0); break;
      case OBJ_AXIS      : drawAxis(0,0,0,1.0);                  break;
      case OBJ_CUBE      : drawCube();                           break;
      case OBJ_PYRAMID   : drawPyramid();                        break;
      case OBJ_SPHERE    : drawSphere( SPHERE_QUALITY );         break;
      case OBJ_INVISIBLE : /*DONT DRAW ANYTHING*/                break;
      case OBJ_QUESTION  : drawQuestion();                       break;
      case OBJ_BBOX      : drawBoundingBox(0,0,0,-1.0,-1.0,-1.0,1.0,1.0,1.0); break;
      default :
       return 0;
      break;
    }
  return 1;
}


unsigned int drawHardcodedModel(unsigned int modelType)
{
   //
   #if USE_GLEW
   fprintf(stderr,"shader mode is under construction please not that things might be invisible.. | drawHardcodedModel(%u)\n",modelType);
     return drawHardcodedModelRaw(modelType);
   #else
   #if DISABLE_GL_CALL_LIST
      drawHardcodedModelRaw(modelType);
    #else
      if ( modelType >= TOTAL_POSSIBLE_MODEL_TYPES )
      {
       fprintf(stderr,"Cannot draw object list for object out of reference");
       return 0;
      }

      if (hardcodedObjlist[modelType]!=0)
        { glCallList(hardcodedObjlist[modelType]); }
      return 1;
    #endif // DISABLE_GL_CALL_LIST
   #endif // OLD_WAY_TO_DRAW
}


int drawConnector(
                  float * posA,
                  float * posB,
                  float * scale ,
                  float * R , float * G , float * B , float * Alpha )
{
 glPushMatrix();
   //fprintf(stderr,"CONNECTOR SIZE %0.2f\n",*scale);
    glLineWidth(*scale);
    glColor3f(*R,*G,*B); //Alpha not used ?
     glBegin(GL_LINES);
       glVertex3f(posA[0],posA[1],posA[2]);
       glVertex3f(posB[0],posB[1],posB[2]);
     glEnd();
 glPopMatrix();
 return 1;
}


#define NORMAL   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define BLUE    "\033[34m"      /* Blue */


int initializeHardcodedCallLists()
{
  fprintf(stderr,"Initializing Hardcoded Call lists\n");
  fprintf(stderr,"Axis color convention is : \n");
  fprintf(stderr,RED "RED   = X \n" NORMAL);
  fprintf(stderr,GREEN "GREEN = Y \n" NORMAL);
  fprintf(stderr,BLUE "BLUE  = Z \n" NORMAL);

  initializeSphere();

  unsigned int i=0;
  for (i=0; i<TOTAL_POSSIBLE_MODEL_TYPES; i++)
  {
    if (i!=OBJ_MODEL)
    {
     glPushAttrib(GL_ALL_ATTRIB_BITS);
	 hardcodedObjlist[i]=glGenLists(1);
     glNewList(hardcodedObjlist[i],GL_COMPILE);
      drawHardcodedModelRaw(i);
     glEndList();
     glPopAttrib();
    }
  }
}
