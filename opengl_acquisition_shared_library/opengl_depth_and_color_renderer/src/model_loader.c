
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

struct Model * loadModel(char * directory,char * modelname)
{
  struct Model * mod = ( struct Model * ) malloc(sizeof(struct Model));
  if ( mod == 0 )  { fprintf(stderr,"Could not allocate enough space for model %s \n",modelname);  return 0; }
  memset(mod , 0 , sizeof(struct Model));

  if ( strstr(modelname,".obj") != 0 )
    {
      mod->type = OBJMODEL;
      mod->model = (struct  OBJ_Model * ) loadObj(directory,modelname);
    }

  if (mod->model ==0 ) { free(mod); return 0 ;}

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

void drawModelAt(struct Model * mod,float x,float y,float z,float heading,float pitch,float roll)
{
 if (mod!=0)
 {
  int NoColor = 0;
  if ( ( mod->colorR==0.482f ) &&
       ( mod->colorG==0.482f ) &&
       ( mod->colorB==0.0f   )  )   { NoColor = 1; }

  glPushMatrix();
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glEnable(GL_NORMALIZE);
  if (mod->nocull) { glDisable(GL_CULL_FACE); }
  glTranslated(x,y,z);
  if ( roll!=0 ) { glRotated(roll,0.0,0.0,1.0); }
  if ( heading!=0 ) { glRotated(heading,0.0,1.0,0.0); }
  if ( pitch!=0 ) { glRotated(pitch,1.0,0.0,0.0); }

   if (NoColor)
      { // MAGIC NO COLOR VALUE :P MEANS NO COLOR SELECTION
        glDisable(GL_COLOR_MATERIAL); //Required for the glMaterial calls to work
      } else
      { if (mod->transparency==0.0)
         {
          //fprintf(stderr,"Only Seting color %0.2f %0.2f %0.2f \n",mod->colorR,mod->colorG,mod->colorB);
          glColor3f(mod->colorR,mod->colorG,mod->colorB);
         } else
         { glEnable(GL_BLEND);			// Turn Blending On
           glBlendFunc(GL_SRC_ALPHA, GL_ONE);
           glColor4f(mod->colorR,mod->colorG,mod->colorB,mod->transparency);
         }
      }




    switch ( mod->type )
    {
      case OBJMODEL :
      {
         GLuint objlist  =  getObjOGLList( ( struct OBJ_Model * ) mod->model);
         if (objlist!=0)
          {
              //We have compiled a list of the triangles for better performance
              glCallList(objlist);
          }  else
          {
              //Just feed the triangles to open gl one by one ( slow )
              drawOBJMesh( ( struct OBJ_Model * ) mod->model);
          }

      }
      break;
    };


       if (mod->transparency!=0) {glDisable(GL_BLEND);  }
       if (  NoColor ) {glEnable(GL_COLOR_MATERIAL);   }
       if (mod->nocull) { glEnable(GL_CULL_FACE); }

  glTranslated(-x,-y,-z);
  glDisable(GL_NORMALIZE);
  glPopMatrix();
} else
{
    fprintf(stderr,"Cannot draw model , it doesnt exist \n");
}
}


void drawModel(struct Model * mod)
{
    if (mod == 0) { fprintf(stderr,"Cannot draw model , it doesnt exist \n"); return ; } //If mod = 0 accesing the fields below will lead in crashing..
    drawModelAt(mod,mod->x,mod->y,mod->z,mod->heading,mod->pitch,mod->roll);
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



int setModelColor(struct Model * mod,float *R,float *G,float *B,float *transparency)
{
 if (mod==0) { return 0; }

 //fprintf(stderr,"Seting color to  %0.2f %0.2f %0.2f trans %0.2f \n",*R,*G,*B,*transparency);
 mod->colorR = *R;
 mod->colorG = *G;
 mod->colorB = *B;
 mod->transparency = *transparency;
 //fprintf(stderr,"Seting color to  %0.2f %0.2f %0.2f trans %0.2f \n",mod->colorR,mod->colorG,mod->colorB,mod->transparency);
 return 1;
}

int drawCube()
{
    glNewList(1, GL_COMPILE_AND_EXECUTE);
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
    glEndList();
    return 1;
}




