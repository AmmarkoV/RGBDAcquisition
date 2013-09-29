#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include "tiledRenderer.h"

#include <stdio.h>

#include "TrajectoryParser/TrajectoryParser.h"
#include "model_loader.h"
#include "scene.h"


enum POS_COORDS
{
    POS_X=0,
    POS_Y,
    POS_Z,
    POS_ANGLEX,
    POS_ANGLEY,
    POS_ANGLEZ,
    POS_COORD_LENGTH
};


int getPhotoshootTile2DCoords(unsigned int column, unsigned int row , double * x2D , double *y2D , double * z2D)
{
      GLint viewport[4];
      GLdouble modelview[16];
      GLdouble projection[16];
      GLdouble posX, posY, posZ=0.0;
      GLdouble winX, winY, winZ=0.0;

      glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
      glGetDoublev( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );

      gluProject( posX, posY, posZ , modelview, projection, viewport, &winX, &winY, &winZ);
      return 0;
}



void setupTiledRendererOGL(float backgroundR,float backgroundG,float backgroundB)
{
  glClearColor(backgroundR,backgroundG,backgroundB,0.0);

  glEnable (GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW );


  glLoadIdentity();
  glRotatef(0,-1.0,0,0); // Peristrofi gyrw apo ton x
  glRotatef(0,0,-1.0,0); // Peristrofi gyrw apo ton y
  glRotatef(0,0,0,-1.0);
  glTranslatef(0,0,0);

}



int tiledRenderer_Render( struct tiledRendererConfiguration * configuration)
{
  unsigned int columns=configuration->columns;
  unsigned int rows=configuration->rows;
  unsigned int objID=configuration->objID;
  float distance=configuration->distance;
  float angleX=configuration->angleX;
  float angleY=configuration->angleY;
  float angleZ=configuration->angleZ;
  float angXVariance=configuration->angXVariance;
  float angYVariance=configuration->angYVariance;
  float angZVariance=configuration->angZVariance;



  struct VirtualStream * scene = (struct VirtualStream *)  configuration->scenePTR;
  struct Model ** models = ( struct Model ** ) configuration->modelPTR;


  fprintf(stderr,"Photoshooting Object %u -> %s \n",objID,scene->object[objID].name);
  fprintf(stderr,"Rows/Cols %u/%u  Distance %0.2f , Angles %0.2f %0.2f %0.2f\n",rows,columns,distance,angleX,angleY,angleZ);
  fprintf(stderr,"Angle Variance %0.2f %0.2f %0.2f\n",angXVariance,angYVariance,angZVariance);


  if (scene!=0) { setupTiledRendererOGL((float)scene->backgroundR,(float)scene->backgroundG,(float)scene->backgroundB); } else
                { setupTiledRendererOGL(0.0,0.0,0.0); }


  if (scene!=0)
    {
       unsigned char noColor=0;
       float posStack[POS_COORD_LENGTH]={0};
       float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;
       unsigned int i=objID;

       struct Model * mod = models[scene->object[i].type];
       float * pos = (float*) &posStack;
         //This is a stupid way of passing stuff to be drawn
         R=1.0f; G=1.0f;  B=1.0f; trans=0.0f; noColor=0;
         getObjectColorsTrans(scene,i,&R,&G,&B,&trans,&noColor);

         setModelColor(mod,&R,&G,&B,&trans,&noColor);
         mod->scale = scene->object[i].scale;


        int x,y,z;


        float OGLUnitWidth=2.5 , OGLUnitHeight =2.4;

        int snapsHorizontal=columns;
        int snapsVertical=rows;


        float posOffsetX = 0;// -4;
        float posOffsetY = 0;//  4;

        float posXBegining= -1*(posOffsetX+(float) snapsHorizontal/2)*OGLUnitWidth;
        float posYBegining= -1*(posOffsetY+(float) snapsVertical/2)*OGLUnitHeight;

        float angXStep = (float)(2*angXVariance)/snapsHorizontal;
        float angYStep = (float)(2*angYVariance)/snapsVertical  ;
        float angZStep=0;

        posStack[POS_Z]= 0 - distance ;

        posStack[POS_ANGLEZ]=angleZ;


      fprintf(stderr,"Drawing starts @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",posXBegining,posYBegining  ,  angleX-angXVariance , angleY-angYVariance , angleZ);



       posStack[POS_ANGLEY]=(float) angleY-angYVariance;
       posStack[POS_Y]=posYBegining;
       for (y=0; y<=snapsVertical; y++)
          {
           posStack[POS_ANGLEY]+=angYStep;
           posStack[POS_Y]+=OGLUnitHeight;


            posStack[POS_ANGLEX]=(float) angleX-angXVariance;
            posStack[POS_X]=posXBegining;
            for (x=0; x<snapsHorizontal; x++)
               {
                 posStack[POS_ANGLEX]+=angXStep;
                 posStack[POS_X]+=OGLUnitWidth;

                   fprintf(stderr,"Drawing model  @ %0.2f %0.2f %0.2f   %0.2f %0.2f %0.2f \n",posStack[POS_X],posStack[POS_Y],posStack[POS_Z],pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]);

                   drawModelAt(mod,pos[POS_X],pos[POS_Y],pos[POS_Z],pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]);
                }
            }
          fprintf(stderr,"Drawing stopped  @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",posStack[POS_X],posStack[POS_Y],pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]);
        }

   glPopMatrix();
  return 1 ;
}
