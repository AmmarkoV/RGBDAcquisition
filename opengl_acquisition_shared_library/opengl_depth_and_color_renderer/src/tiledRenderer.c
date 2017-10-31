#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include "tiledRenderer.h"

#include <stdio.h>
#include "../../../tools/AmMatrix/matrixProject.h"

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
      int viewport[4];
      float modelview[16];
      float projection[16];
      float win[3]={0};

      glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
      glGetFloatv( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );

      float x3D , y3D , z3D , angleX , angleY , angleZ;
      tiledRenderer_get3DCenterForTile(trConf , column, row ,&x3D , &y3D , &z3D , &angleX , &angleY , &angleZ);
      float posX = x3D , posY = y3D , posZ = z3D;

      _glhProjectf( posX, posY, posZ , modelview, projection, viewport, win);

      fprintf(stderr,"Column/Row %u/%u ( %0.2f,%0.2f,%0.2f ) -> %0.2f %0.2f %0.2f\n",column,row , x3D , y3D , z3D , win[0] , win[1] , win[2]);

      //struct tiledRendererConfiguration * trConfPTR = (struct tiledRendererConfiguration *) trConf;
      *x2D = win[0];
      *y2D = win[1];
      *z2D = win[2];
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

        trConf->op.OGLUnitWidth =widthToUse+1.0;
        trConf->op.OGLUnitHeight=widthToUse+1.0;

        trConf->op.snapsHorizontal=trConf->columns;
        trConf->op.snapsVertical  =trConf->rows;

        float halfSnapsHorizontal = (float) trConf->columns/2;
        float halfSnapsVertical   = (float) trConf->rows/2;

        trConf->op.posOffsetX = -0.5;
        trConf->op.posOffsetY = -0.5;

        trConf->op.posXBegining= -1*(trConf->op.posOffsetX+halfSnapsHorizontal)*trConf->op.OGLUnitWidth;
        trConf->op.posYBegining= -1*(trConf->op.posOffsetY+halfSnapsVertical)  *trConf->op.OGLUnitHeight;

        trConf->op.angXStep = (float)(2*trConf->angXVariance)/trConf->op.snapsHorizontal;
        trConf->op.angYStep = (float)(2*trConf->angYVariance)/trConf->op.snapsVertical  ;
        trConf->op.angZStep=0;

        fprintf(stderr,"Drawing starts @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",trConf->op.posXBegining,trConf->op.posYBegining  ,  trConf->angleX-trConf->angXVariance , trConf->angleY-trConf->angYVariance , trConf->angleZ);
        fprintf(stderr,"Tile Size selected is %u,%u -> %u,%u \n",trConf->op.snapsHorizontal , trConf->op.snapsVertical , trConf->columns , trConf->rows);

 return 1;
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


  fprintf(stderr,"setupTiledRendererOGL done \n");
  if (scene!=0)
    {
       unsigned char noColor=0;
       float posStack[POS_COORD_LENGTH]={0};
       float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;

       unsigned int i=trConf->objID;
       fprintf(stderr,"Accessing models (%u).. \n",scene->object[i].type);
       struct Model * mod = models[scene->object[i].type];
       if (mod==0) { fprintf(stderr,"Model not allocated.. \n"); return 0;}
       float * pos = (float*) &posStack;

       fprintf(stderr,"getObjectColorsTrans Colors.. \n");
       //This is a stupid way of passing stuff to be drawn
       R=1.0f; G=1.0f;  B=1.0f; trans=0.0f; noColor=0;
       getObjectColorsTrans(scene,i,&R,&G,&B,&trans,&noColor);

       fprintf(stderr,"Setting Colors.. ");
       setModelColor(mod,&R,&G,&B,&trans,&noColor);
       fprintf(stderr,"done\n");
       mod->scaleX = scene->object[i].scaleX;
       mod->scaleY = scene->object[i].scaleY;
       mod->scaleZ = scene->object[i].scaleZ;

/*
       pos[POS_X]=0; pos[POS_Y]=0; pos[POS_Z]=-30;
                   drawModelAt(
                                mod,
                                pos[POS_X],pos[POS_Y],pos[POS_Z],
                                pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ]
                              );
*/
        unsigned int x,y;

        fprintf(stderr,"Calculating loops for tiled renderer..\n");
        tiledRenderer_CalculateLoops(trConf);

       for (y=0; y<trConf->op.snapsVertical; y++)
          {
            for (x=0; x<trConf->op.snapsHorizontal; x++)
               {
                   tiledRenderer_get3DCenterForTile( trConf , x , y ,
                                                     &pos[POS_X],&pos[POS_Y],&pos[POS_Z],
                                                     &pos[POS_ANGLEX],&pos[POS_ANGLEY],&pos[POS_ANGLEZ]);


                   fprintf(stderr,"Draw %u,%u @ %0.2f %0.2f %0.2f\n",x,y,pos[POS_X],pos[POS_Y],pos[POS_Z]);
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





