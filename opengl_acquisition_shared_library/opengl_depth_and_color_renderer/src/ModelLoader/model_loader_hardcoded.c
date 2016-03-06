
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <math.h>

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "model_loader.h"
#include "model_loader_hardcoded.h"
#include "model_loader_obj.h"
#include "../tools.h"

#define DISABLE_GL_CALL_LIST 0
#if DISABLE_GL_CALL_LIST
 #warning "Please note that glCallList is disabled and that has a really bad effect on graphics card performance"
#endif // DISABLE_GL_CALL_LIST

#define SPHERE_QUALITY 10 /*100 is good quality*/

#define PIE 3.14159265358979323846
#define degreeToRadOLD(deg) (deg)*(PIE/180)



#define USE_QUESTIONMARK_FOR_FAILED_LOADED_MODELS 1

GLuint hardcodedObjlist[TOTAL_POSSIBLE_MODEL_TYPES]={0};


#define U 0.5

float cubeCoords[]={ //X  Y  Z       W
                      //Near
                     -U, -U, -U,    1.0,  // bottom left
                     -U,  U, -U,    1.0,  // top left
                      U,  U, -U,    1.0,  // top right

                     +U, -U, -U,    1.0,  // bottom right
                     -U, -U, -U,    1.0,// bottom left corner
                     +U, +U, -U,    1.0,// top left corner

                     //Far
                     -U, -U, U,    1.0,  // bottom left
                     -U,  U, U,    1.0,  // top left
                      U,  U, U,    1.0,  // top right

                     +U, -U, U,    1.0,  // bottom right
                     -U, -U, U,    1.0,// bottom left corner
                     +U, +U, U,    1.0,// top left corner

                     //Left
                     -U, -U, -U,    1.0,  // bottom left
                     -U,  U, -U,    1.0,  // top left
                     -U,  U,  U,    1.0,  // top right

                     -U, -U, -U,    1.0,  // bottom right
                     -U, +U, +U,    1.0,// bottom left corner
                     -U, -U,  U,    1.0,// top left corner

                     //Right
                      U, -U, -U,    1.0,  // bottom left
                      U,  U, -U,    1.0,  // top left
                      U,  U,  U,    1.0,  // top right

                      U, -U, -U,    1.0,  // bottom right
                      U, +U, +U,    1.0,// bottom left corner
                      U, -U,  U,    1.0,// top left corner


                     //Up
                     -U,  U, -U,    1.0,  // bottom left
                     -U,  U,  U,    1.0,  // top left
                      U,  U,  U,    1.0,  // top right

                     -U,  U, -U,    1.0,  // bottom right
                      U,  U,  U,    1.0,// bottom left corner
                      U,  U, -U,    1.0,// top left corner


                     //Bottom
                     -U, -U, -U,    1.0,  // bottom left
                     -U, -U,  U,    1.0,  // top left
                      U, -U,  U,    1.0,  // top right

                     -U, -U, -U,    1.0,  // bottom right
                      U, -U,  U,    1.0,// bottom left corner
                      U, -U, -U,    1.0,// top left corner

                  };




float pyramidCoords[]={ //X  Y  Z       W

                     //Near
                     -U, -U, -U,    1.0,  // bottom left
                      0, U,  0,    1.0,  // top left
                      U, -U, -U,    1.0,  // top right

                     //Far
                     -U, -U,  U,    1.0,  // bottom left
                      0, U,  0,    1.0,  // top left
                      U, -U,  U,    1.0,  // top right

                     //Left
                     -U, -U, -U,    1.0,  // bottom left
                      0, U,  0,    1.0,  // top left
                     -U, -U,  U,    1.0,  // top right

                     //Right
                      U, -U, -U,    1.0,  // bottom left
                      0, U,  0,    1.0,  // top left
                      U, -U,  U,    1.0,  // top right


                     //Bottom
                     -U, -U, -U,    1.0,  // bottom left
                     -U, -U,  U,    1.0,  // top left
                      U, -U,  U,    1.0,  // top right

                     -U, -U, -U,    1.0,  // bottom right
                      U, -U,  U,    1.0,// bottom left corner
                      U, -U, -U,    1.0,// top left corner

                    };







int drawAxis(float x, float y , float z, float scale)
{
 glLineWidth(6.0);
 glBegin(GL_LINES);
  glColor3f(1.0,0.0,0.0); glVertex3f(x,y,z); glVertex3f(x+scale,y,z);
  glColor3f(0.0,1.0,0.0); glVertex3f(x,y,z); glVertex3f(x,y+scale,z);
  glColor3f(0.0,0.0,1.0); glVertex3f(x,y,z); glVertex3f(x,y,z+scale);
 glEnd();
 glLineWidth(1.0);
 return 1;
}

int drawObjPlane(float x,float y,float z,float dimension)
{
   // glNewList(1, GL_COMPILE_AND_EXECUTE);
    /* front face */
    glBegin(GL_QUADS);
      glNormal3f(0.0,1.0,0.0);
      glVertex3f(x-dimension, 0.0, z-dimension);
      glVertex3f(x+dimension, 0.0, z-dimension);
      glVertex3f(x+dimension, 0.0, z+dimension);
      glVertex3f(x-dimension, 0.0, z+dimension);
    glEnd();
    //glEndList();
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

//    double r=1.0;
    int lats=quality;
    int longs=quality;
  //---------------
    int i, j;
    for(i = 0; i <= lats; i++)
    {
       double lat0 = M_PI * (-0.5 + (double) (i - 1) / lats);
       double z0  = sin(lat0);
       double zr0 =  cos(lat0);

       double lat1 = M_PI * (-0.5 + (double) i / lats);
       double z1 = sin(lat1);
       double zr1 = cos(lat1);

       glBegin(GL_QUAD_STRIP);
       for(j = 0; j <= longs; j++)
        {
           double lng = 2 * M_PI * (double) (j - 1) / longs;
           double x = cos(lng);
           double y = sin(lng);

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
  //Todo draw a question mark here , this is an axis actually :P
  float x=0,y=0,z=0,scale=1.0;
  glLineWidth(6.0);
  glBegin(GL_LINES);
   glColor3f(1.0,0.0,0.0); glVertex3f(x,y,z); glVertex3f(x+scale,y,z);
   glColor3f(0.0,1.0,0.0); glVertex3f(x,y,z); glVertex3f(x,y+scale,z);
   glColor3f(0.0,0.0,1.0); glVertex3f(x,y,z); glVertex3f(x,y,z+scale);
  glEnd();
  glLineWidth(1.0);

  //drawSphere(30);
 return 1;
}

int drawBoundingBox(float x,float y,float z ,float minX,float minY,float minZ,float maxX,float maxY,float maxZ)
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



int drawGenericTriangleMesh(float * coords , unsigned int coordLength)
{
    glBegin(GL_TRIANGLES);
      unsigned int i=0;
      for (i=0; i<coordLength; i++)
        { glVertex3f(coords[i*4+0],coords[i*4+1], coords[i*4+2] ); }
    glEnd();
    return 1;
}

int drawCube()
{
    return drawGenericTriangleMesh(cubeCoords , sizeof(cubeCoords)/(4*sizeof(float)) );
}

void drawPyramid()
{
    return drawGenericTriangleMesh(pyramidCoords , sizeof(pyramidCoords)/(4*sizeof(float)) );
}

unsigned int isModelnameAHardcodedModel(const char * modelname,unsigned int * itIsAHardcodedModel)
{
  *itIsAHardcodedModel=1;
  unsigned int modType=0;
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
      case OBJ_PLANE :     drawObjPlane(0,0,0, 0.5);             break;
      case OBJ_GRIDPLANE : drawGridPlane( 0.0 , 0.0 , 0.0, 1.0); break;
      case OBJ_AXIS :      drawAxis(0,0,0,1.0);                  break;
      case OBJ_CUBE :      drawCube();                           break;
      case OBJ_PYRAMID :   drawPyramid();                        break;
      case OBJ_SPHERE  :   drawSphere( SPHERE_QUALITY );         break;
      case OBJ_INVISIBLE : /*DONT DRAW ANYTHING*/                break;
      case OBJ_QUESTION  : drawQuestion();                       break;
      case OBJ_BBOX :  drawBoundingBox(0,0,0,-1.0,-1.0,-1.0,1.0,1.0,1.0); break;
      default :
       return 0;
      break;
    }
  return 1;
}


unsigned int drawHardcodedModel(unsigned int modelType)
{
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
    #endif // DISABLE_GL_CALL_LIST
  return 1;
}


int drawConnector(
                  float * posA,
                  float * posB,
                  float * scale ,
                  unsigned char R , unsigned char G , unsigned char B , unsigned char Alpha )
{
 glPushMatrix();
    glLineWidth(*scale);
    glColor3f(R,G,B); //Alpha not used ?
     glBegin(GL_LINES);
       glVertex3f(posA[0],posA[1],posA[2]);
       glVertex3f(posB[0],posB[1],posB[2]);
     glEnd();
 glPopMatrix();
 return 1;
}

int initializeHardcodedCallLists()
{
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
