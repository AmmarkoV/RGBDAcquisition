#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include "tiledRenderer.h"

#include <stdio.h>

#include "TrajectoryParser/TrajectoryParser.h"
#include "ModelLoader/model_loader.h"
#include "scene.h"



int tiledRenderer_get3DCenterForTile(struct tiledRendererConfiguration * trConf , unsigned int column,unsigned int row ,
                                     float * x3D , float * y3D , float * z3D ,
                                     float * angleX , float * angleY , float * angleZ)
{
  *x3D = trConf->op.posXBegining + (column * trConf->op.OGLUnitWidth);
  *y3D = trConf->op.posYBegining + (row * trConf->op.OGLUnitHeight);
  *z3D = 0 - trConf->distance ;

  *angleX = trConf->angleX - trConf->angXVariance + (trConf->op.angXStep *  column);
  *angleY = trConf->angleY - trConf->angYVariance  + (trConf->op.angYStep *  row);
  *angleZ = trConf->angleZ;

  return 1;
}


int tiledRenderer_get2DCenter(void * trConf ,
                              unsigned int column, unsigned int row ,
                              float * x2D , float *y2D , float * z2D)
{
      GLint viewport[4];
      GLdouble modelview[16];
      GLdouble projection[16];


      float x3D , y3D , z3D , angleX , angleY , angleZ;
      tiledRenderer_get3DCenterForTile(trConf , column, row ,&x3D , &y3D , &z3D , &angleX , &angleY , &angleZ);
      GLdouble posX = x3D , posY = y3D , posZ = z3D;
      GLdouble winX, winY, winZ=0.0;

      glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
      glGetDoublev( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );

      gluProject( posX, posY, posZ , modelview, projection, viewport, &winX, &winY, &winZ);

      //fprintf(stderr,"Column/Row %u/%u ( %0.2f,%0.2f,%0.2f ) -> %0.2f %0.2f %0.2f\n",column, x3D , y3D , z3D , row , winX , winY , winZ);

      struct tiledRendererConfiguration * trConfPTR = (struct tiledRendererConfiguration *) trConf;
      *x2D = winX;
      *y2D = winY;
      *z2D = winZ;
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




int tiledRenderer_CalculateLoops( struct tiledRendererConfiguration * trConf)
{
        struct VirtualStream * scene = (struct VirtualStream *)  trConf->scenePTR;
        struct Model ** models = ( struct Model ** ) trConf->modelPTR;
        unsigned int i=trConf->objID;
        struct Model * mod = models[scene->object[i].type];

        float sizeX , sizeY , sizeZ;
        getModel3dSize(mod , &sizeX , &sizeY , &sizeZ );

        float widthToUse = sizeX;
        if (sizeY>widthToUse) { widthToUse = sizeY; }
        if (sizeZ>widthToUse) { widthToUse = sizeY; }

        trConf->op.OGLUnitWidth=widthToUse+2.0;
        trConf->op.OGLUnitHeight=widthToUse+2.0;

        trConf->op.snapsHorizontal=trConf->columns;
        trConf->op.snapsVertical=trConf->rows;

        trConf->op.posOffsetX = 0;
        trConf->op.posOffsetY = 0;

        trConf->op.posXBegining= -1*(trConf->op.posOffsetX+(float) trConf->op.snapsHorizontal/2)*trConf->op.OGLUnitWidth;
        trConf->op.posYBegining= -1*(trConf->op.posOffsetY+(float) trConf->op.snapsVertical/2)  *trConf->op.OGLUnitHeight;

        trConf->op.angXStep = (float)(2*trConf->angXVariance)/trConf->op.snapsHorizontal;
        trConf->op.angYStep = (float)(2*trConf->angYVariance)/trConf->op.snapsVertical  ;
        trConf->op.angZStep=0;

      fprintf(stderr,"Drawing starts @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",trConf->op.posXBegining,trConf->op.posYBegining  ,  trConf->angleX-trConf->angXVariance , trConf->angleY-trConf->angYVariance , trConf->angleZ);
}




int tiledRenderer_Render( struct tiledRendererConfiguration * trConf)
{
  if (trConf==0) { fprintf(stderr,"Could not render with null configuration\n"); return 0; }
  if (trConf->scenePTR==0) { fprintf(stderr,"Could not render with null scene\n"); return 0; }
  if (trConf->modelPTR==0) { fprintf(stderr,"Could not render with null model\n"); return 0; }

  struct VirtualStream * scene = (struct VirtualStream *)  trConf->scenePTR;
  struct Model ** models = ( struct Model ** ) trConf->modelPTR;




  fprintf(stderr,"Photoshooting Object %u -> %s \n",trConf->objID,scene->object[trConf->objID].name);
  fprintf(stderr,"Rows/Cols %u/%u  Distance %0.2f , Angles %0.2f %0.2f %0.2f\n",trConf->rows,trConf->columns,trConf->distance,trConf->angleX,trConf->angleY,trConf->angleZ);
  fprintf(stderr,"Angle Variance %0.2f %0.2f %0.2f\n",trConf->angXVariance,trConf->angYVariance,trConf->angZVariance);


  if (scene!=0) { setupTiledRendererOGL((float)scene->backgroundR,(float)scene->backgroundG,(float)scene->backgroundB); } else
                { setupTiledRendererOGL(0.0,0.0,0.0); }


  if (scene!=0)
    {
       unsigned char noColor=0;
       float posStack[POS_COORD_LENGTH]={0};
       float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;
       unsigned int i=trConf->objID;

       struct Model * mod = models[scene->object[i].type];
       float * pos = (float*) &posStack;
         //This is a stupid way of passing stuff to be drawn
         R=1.0f; G=1.0f;  B=1.0f; trans=0.0f; noColor=0;
         getObjectColorsTrans(scene,i,&R,&G,&B,&trans,&noColor);

         setModelColor(mod,&R,&G,&B,&trans,&noColor);
         mod->scaleX = scene->object[i].scaleX;
         mod->scaleY = scene->object[i].scaleY;
         mod->scaleZ = scene->object[i].scaleZ;

        int x,y,z;

        tiledRenderer_CalculateLoops(trConf);

       for (y=0; y<=trConf->op.snapsVertical; y++)
          {
            for (x=0; x<trConf->op.snapsHorizontal; x++)
               {
                   tiledRenderer_get3DCenterForTile( trConf , x , y ,
                                                     &pos[POS_X],&pos[POS_Y],&pos[POS_Z],
                                                     &pos[POS_ANGLEX],&pos[POS_ANGLEY],&pos[POS_ANGLEZ]);


                   drawModelAt(
                                mod,
                                pos[POS_X],pos[POS_Y],pos[POS_Z],
                                pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]
                              );
                }
            }
          fprintf(stderr,"Drawing stopped  @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",posStack[POS_X],posStack[POS_Y],pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]);
        }

   glPopMatrix();
  return 1 ;
}





