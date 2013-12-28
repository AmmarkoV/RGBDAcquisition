
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "model_loader.h"
#include "model_loader_obj.h"
#include "tools.h"


#define PIE 3.14159265358979323846
#define degreeToRadOLD(deg) (deg)*(PIE/180)

const GLfloat defaultAmbient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat defaultDiffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat defaultSpecular[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
const GLfloat defaultShininess[] = { 5.0f };



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


int drawCube()
{
   // glNewList(1, GL_COMPILE_AND_EXECUTE);
    /* front face */
    glBegin(GL_QUADS);
      glColor3f(0.0, 0.7, 0.1);  /* green */
      glVertex3f(-1.0, 1.0, 1.0);
      glVertex3f(1.0, 1.0, 1.0);
      glVertex3f(1.0, -1.0, 1.0);
      glVertex3f(-1.0, -1.0, 1.0);

      /* back face */
      glColor3f(0.9, 1.0, 0.0);  /* yellow */
      glVertex3f(-1.0, 1.0, -1.0);
      glVertex3f(1.0, 1.0, -1.0);
      glVertex3f(1.0, -1.0, -1.0);
      glVertex3f(-1.0, -1.0, -1.0);

      /* top side face */
      glColor3f(0.2, 0.2, 1.0);  /* blue */
      glVertex3f(-1.0, 1.0, 1.0);
      glVertex3f(1.0, 1.0, 1.0);
      glVertex3f(1.0, 1.0, -1.0);
      glVertex3f(-1.0, 1.0, -1.0);

      /* bottom side face */
      glColor3f(0.7, 0.0, 0.1);  /* red */
      glVertex3f(-1.0, -1.0, 1.0);
      glVertex3f(1.0, -1.0, 1.0);
      glVertex3f(1.0, -1.0, -1.0);
      glVertex3f(-1.0, -1.0, -1.0);
    glEnd();
   // glEndList();
    return 1;
}



struct Model * loadModel(char * directory,char * modelname)
{
  if ( (directory==0) || (modelname==0) )
  {
    fprintf(stderr,"loadModel failing , no modelname given");
    return 0;
  }

  struct Model * mod = ( struct Model * ) malloc(sizeof(struct Model));
  if ( mod == 0 )  { fprintf(stderr,"Could not allocate enough space for model %s \n",modelname);  return 0; }
  memset(mod , 0 , sizeof(struct Model));

  if ( strcmp(modelname,"plane") == 0 ) {  mod->type = OBJ_PLANE; mod->model = 0; }  else
  if ( strcmp(modelname,"grid") == 0 ) {  mod->type = OBJ_GRIDPLANE; mod->model = 0; }  else
  if ( strcmp(modelname,"cube") == 0 ) {  mod->type = OBJ_CUBE; mod->model = 0; }  else
  if ( strcmp(modelname,"axis") == 0 ) {  mod->type = OBJ_AXIS; mod->model = 0; }  else
  if ( strstr(modelname,".obj") != 0 )
    {
      mod->type = OBJMODEL;
      mod->model = (struct  OBJ_Model * ) loadObj(directory,modelname);
      if (mod->model ==0 ) { free(mod); return 0 ;}
    } else
    {
      fprintf(stderr,"Could not understand how to load object %s \n",modelname);
      fprintf(stderr,"Searched in directory %s \n",directory);
      fprintf(stderr,"Object %s was also not one of the hardcoded shapes\n",modelname);
      if (mod->model ==0 ) { free(mod); return 0 ;}
    }

  return mod;
}

void unloadModel(struct Model * mod)
{
   if (mod == 0 ) { return ; }

    switch ( mod->type )
    {
      case OBJMODEL :
          unloadObj( (struct  OBJ_Model * ) mod->model);
      break;
    };
}

int drawModelAt(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll)
{
 if (mod!=0)
 {
  glPushMatrix();
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glEnable(GL_NORMALIZE);
  /*If scale factors other than 1 are applied to the modelview matrix
            and lighting is enabled, lighting often appears wrong.
            In that case, enable automatic normalization of normals by
            calling glEnable with the argument GL_NORMALIZE.*/

  if (mod->nocull) { glDisable(GL_CULL_FACE); }
  if (mod->scale!=1.0) {
                         glScalef(mod->scale,mod->scale,mod->scale);
                         int err=glGetError();
                         if (err !=  GL_NO_ERROR/*0*/ ) { fprintf(stderr,"Could not scale ( error %u ) :(\n",err); }
                         fprintf(stderr,"Scaling model by %0.2f\n",mod->scale);
                       }
  glTranslatef(x,y,z);
  if ( roll!=0 ) { glRotatef(roll,0.0,0.0,1.0); }
  if ( heading!=0 ) { glRotatef(heading,0.0,1.0,0.0); }
  if ( pitch!=0 ) { glRotatef(pitch,1.0,0.0,0.0); }

       // MAGIC NO COLOR VALUE :P MEANS NO COLOR SELECTION
      if (mod->nocolor!=0)  { glDisable(GL_COLOR_MATERIAL);   } else
      {//We Have a color to set
        glEnable(GL_COLOR);
        glEnable(GL_COLOR_MATERIAL);

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,    defaultAmbient);
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,    defaultDiffuse);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,   defaultSpecular);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS,   defaultShininess);

        if (mod->transparency==0.0)
        {
          //fprintf(stderr,"SET RGB(%0.2f,%0.2f,%0.2f)\n",mod->colorR, mod->colorG, mod->colorB);
          glColor3f(mod->colorR,mod->colorG,mod->colorB); }
        else
        { glEnable(GL_BLEND);			// Turn Blending On
          glBlendFunc(GL_SRC_ALPHA, GL_ONE);
          //fprintf(stderr,"SET RGBA(%0.2f,%0.2f,%0.2f,%0.2f)\n",mod->colorR, mod->colorG, mod->colorB, mod->transparency);
          glColor4f(mod->colorR,mod->colorG,mod->colorB,mod->transparency);
        }
      }

    //fprintf(stderr,"Drawing RGB(%0.2f,%0.2f,%0.2f) , Transparency %0.2f , ColorDisabled %u\n",mod->colorR, mod->colorG, mod->colorB, mod->transparency,mod->nocolor );

    switch ( mod->type )
    {
      case OBJ_PLANE :     drawObjPlane(0,0,0, 0.5);             break;
      case OBJ_GRIDPLANE : drawGridPlane( 0.0 , 0.0 , 0.0, 1.0); break;
      case OBJ_AXIS :      drawAxis(0,0,0,1.0);                  break;
      case OBJ_CUBE :      drawCube();                           break;
      case OBJMODEL :
      {
         if (mod->model!=0)
         {
           //A model has been created , and it can be served
           GLuint objlist  =  getObjOGLList( ( struct OBJ_Model * ) mod->model);
           if (objlist!=0)
             { //We have compiled a list of the triangles for better performance
               glCallList(objlist);
             }  else
             { //Just feed the triangles to open gl one by one ( slow )
               drawOBJMesh( ( struct OBJ_Model * ) mod->model);
             }
         } else
         { fprintf(stderr,"Could not draw unspecified model\n"); }
         glDisable(GL_TEXTURE_2D); //TODO : <-- change drawOBJMesh , Calllist so that they dont leave textures on! :P
      }
      break;
    };

  if (mod->transparency!=0.0) {glDisable(GL_BLEND);  }
  if (mod->nocolor) {glEnable(GL_COLOR); glEnable(GL_COLOR_MATERIAL); }
  if (mod->nocull)  {glEnable(GL_CULL_FACE); }

  glTranslatef(-x,-y,-z);
  //glDisable(GL_NORMALIZE);
  glPopMatrix();
} else
{
    fprintf(stderr,"Cannot draw model at position %0.2f %0.2f %0.2f , it doesnt exist \n",x,y,z);
    return 0;
}
 return 1;
}


int drawModel(struct Model * mod)
{
    if (mod == 0) { fprintf(stderr,"Cannot draw model , it doesnt exist \n"); return 0; } //If mod = 0 accesing the fields below will lead in crashing..
    return drawModelAt(mod,mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
}


int addToModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll)
{
  if (mod==0) { return 0; }
  mod->x+=x; mod->y+=y; mod->z+=z;

  mod->heading+=heading; mod->pitch+=pitch; mod->roll+=roll;
  fprintf(stderr,"Model(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
  return 1;
}

int addToModelCoordinatesNoSTACK(struct Model * mod,float *x,float *y,float *z,float *heading,float *pitch,float *roll)
{
  if (mod==0) { return 0; }
  mod->x+=*x; mod->y+=*y; mod->z+=*z;

  mod->heading+=*heading; mod->pitch+=*pitch; mod->roll+=*roll;
  fprintf(stderr,"Model(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
  return 1;
}

int setModelCoordinates(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll)
{
  if (mod==0) { return 0; }
  fprintf(stderr,"Model SET Got params(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",x,y,z,heading,pitch,roll);

  mod->x=x; mod->y=y; mod->z=z;

  mod->heading=heading; mod->pitch=pitch; mod->roll=roll;
  fprintf(stderr,"Model SET (%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
  return 1;
}

int setModelCoordinatesNoSTACK(struct Model * mod,float * x,float* y,float *z,float *heading,float *pitch,float* roll)
{
  if (mod==0) { return 0; }
  //fprintf(stderr,"Model SET NoSTACK Got params(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",*x,*y,*z,*heading,*pitch,*roll);

  mod->x=*x; mod->y=*y; mod->z=*z;

  mod->heading=*heading; mod->pitch=*pitch; mod->roll=*roll;
  //fprintf(stderr,"Model SET NoSTACK (%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
  return 1;
}



int setModelColor(struct Model * mod,float *R,float *G,float *B,float *transparency,unsigned char * noColor)
{
 if (mod==0) { return 0; }

 //fprintf(stderr,"Seting color to  %0.2f %0.2f %0.2f trans %0.2f \n",*R,*G,*B,*transparency);
 mod->colorR = *R;
 mod->colorG = *G;
 mod->colorB = *B;
 mod->transparency = *transparency;
 mod->nocolor = *noColor;
 //fprintf(stderr,"Seting color to  %0.2f %0.2f %0.2f trans %0.2f \n",mod->colorR,mod->colorG,mod->colorB,mod->transparency);
 return 1;
}

int getModelBBox(struct Model *mod , float * minX,  float * minY , float * minZ , float * maxX , float * maxY , float * maxZ)
{
 struct OBJ_Model * objMod = (struct OBJ_Model * ) mod->model;
 * minX = objMod->boundBox.min.x;
 * minY = objMod->boundBox.min.y;
 * minZ = objMod->boundBox.min.z;
 * maxX = objMod->boundBox.max.x;
 * maxY = objMod->boundBox.max.y;
 * maxZ = objMod->boundBox.max.z;
 return 1;
}


int getModel3dSize(struct Model *mod , float * sizeX , float * sizeY , float * sizeZ )
{
  float minX,minY,minZ,maxX,maxY,maxZ;
  getModelBBox(mod,&minX,&minY,&minZ,&maxX,&maxY,&maxZ);

  *sizeX = maxX - minX;
  *sizeY = maxY - minY;
  *sizeZ = maxZ - minZ;

 return 1;
}


