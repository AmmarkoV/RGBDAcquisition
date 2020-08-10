#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */
#include "tiledRenderer.h"

#include <stdio.h>
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../../tools/AmMatrix/matrixCalculations.h"

#include "../TrajectoryParser/TrajectoryParser.h"

#include "../ModelLoader/model_loader.h"


#include "../Scene/scene.h"
#include "../Tools/tools.h"

int tiledRenderer_get3DCenterForTile(struct tiledRendererConfiguration * trConf , unsigned int column,unsigned int row ,
                                     float * x3D , float * y3D , float * z3D ,
                                     float * angleX , float * angleY , float * angleZ)
{
  *x3D = 0;//trConf->op.posXBegining + (column * trConf->op.OGLUnitWidth);
  *y3D = 0;//trConf->op.posYBegining + (row * trConf->op.OGLUnitHeight);
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
      float pos3DF[3];
      pos3DF[0]=x3D;
      pos3DF[1]=y3D;
      pos3DF[2]=z3D;

      _glhProjectf(pos3DF, modelview, projection, viewport, win);

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
        struct Model ** models = ( struct Model ** ) trConf->modelStoragePTR;
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

        trConf->op.angXStep = (float)(trConf->angXVariance)/trConf->op.snapsHorizontal;
        trConf->op.angYStep = (float)(trConf->angYVariance)/trConf->op.snapsVertical  ;
        trConf->op.angZStep=0;

        fprintf(stderr,"Drawing starts @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",trConf->op.posXBegining,trConf->op.posYBegining  ,  trConf->angleX-trConf->angXVariance , trConf->angleY-trConf->angYVariance , trConf->angleZ);
        fprintf(stderr,"Tile Size selected is %u,%u -> %u,%u \n",trConf->op.snapsHorizontal , trConf->op.snapsVertical , trConf->columns , trConf->rows);

 return 1;
}

int tiledRenderer_Render( struct tiledRendererConfiguration * trConf)
{
  if (trConf==0) { fprintf(stderr,"Could not render with null configuration\n"); return 0; }
  if (trConf->scenePTR==0) { fprintf(stderr,"Could not render with null scene\n"); return 0; }
  if (trConf->modelStoragePTR==0) { fprintf(stderr,"Could not render with null model\n"); return 0; }

  struct VirtualStream * scene = (struct VirtualStream *)  trConf->scenePTR;

  struct ModelList * modelstorage = ( struct ModelList* ) trConf->modelStoragePTR;
  if (modelstorage->models==0) { fprintf(stderr,"ModelList not properly allocated..\n"); return 0; }

  if (scene!=0) 
    {
       if (scene->object==0) { fprintf(stderr,"Object List not properly allocated..\n"); return 0; }

       fprintf(stderr,"Photoshooting Object %u -> %s \n",trConf->objID,scene->object[trConf->objID].name);
       fprintf(stderr,"Rows/Cols %u/%u  Distance %0.2f , Angles %0.2f %0.2f %0.2f\n",trConf->rows,trConf->columns,trConf->distance,trConf->angleX,trConf->angleY,trConf->angleZ);
       fprintf(stderr,"Angle Variance %0.2f %0.2f %0.2f\n",trConf->angXVariance,trConf->angYVariance,trConf->angZVariance);
 
        
       setupTiledRendererOGL((float)scene->backgroundR,(float)scene->backgroundG,(float)scene->backgroundB);
       unsigned char noColor=0;
       float posStack[POS_COORD_LENGTH]={0};
       float R=1.0f , G=1.0f ,  B=0.0f , trans=0.0f;

       unsigned int i=trConf->objID;
       fprintf(stderr,"Accessing models (%u).. \n",scene->object[i].type);

       fprintf(stderr,"I think i broke the photoshooting here : ");
       if (scene->object[i].type>=modelstorage->currentNumberOfModels) { fprintf(stderr,"Model not allocated.. \n"); return 0; }

       struct Model * mod = &modelstorage->models[scene->object[i].type];
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


       fprintf(stderr,"Will try to fit %ux%u tiles to a %ux%u area\n",trConf->op.snapsHorizontal,trConf->op.snapsVertical,WIDTH,HEIGHT);
        unsigned int tileWidth  =  (unsigned int) WIDTH/trConf->op.snapsHorizontal;
        unsigned int tileHeight  = (unsigned int) HEIGHT/trConf->op.snapsVertical;
       fprintf(stderr,"Tile sizes will be %ux%u\n",tileWidth,tileHeight);


       pos[POS_Z]=0-trConf->distance;

       unsigned long started=GetTickCountMilliseconds();
       for (y=0; y<trConf->op.snapsVertical; y++)
          {
            for (x=0; x<trConf->op.snapsHorizontal; x++)
               {
                   glViewport (x*tileWidth, y*tileHeight, tileWidth, tileHeight);
                   glMatrixMode (GL_PROJECTION);                       // Select The Projection Matrix
                   glLoadIdentity ();                          // Reset The Projection Matrix
                   // Set Up Perspective Mode To Fit 1/4 The Screen (Size Of A Viewport)



                   #define USE_OUR_OWN 1

                   #if USE_OUR_OWN
                    float matrixF[16];
                    gldPerspective(
                                   matrixF,
                                   85.0,
                                   (float) tileWidth/tileHeight,
                                   0.1,
                                   13500.0
                                  );
                    glMultMatrixf(matrixF);
                   #else
                    gluPerspective( 85.0, (GLfloat)(tileWidth)/(GLfloat)(tileHeight), 0.1f, 13500.0 );
                   #endif // USE_OUR_OWN


                   glMatrixMode(GL_MODELVIEW );



                   tiledRenderer_get3DCenterForTile( trConf , x , y ,
                                                     &pos[POS_X],&pos[POS_Y],&pos[POS_Z],
                                                     &pos[POS_ANGLEX],&pos[POS_ANGLEY],&pos[POS_ANGLEZ]);
                   //fprintf(stderr,"Draw %u,%u @ %0.2f %0.2f %0.2f\n",x,y,pos[POS_X],pos[POS_Y],pos[POS_Z]);
                   drawModelAt(
                                mod,
                                pos[POS_X],pos[POS_Y],pos[POS_Z],
                                pos[POS_ANGLEX],pos[POS_ANGLEY],pos[POS_ANGLEZ],
                                mod->rotationOrder
                              );
                }
            }
      unsigned long now=GetTickCountMilliseconds();
      unsigned long elapsedTime=now-started;
      if (elapsedTime==0) { elapsedTime=1; }
      float lastFramerate = (float) 1000/(elapsedTime);

      fprintf(stderr,"Framerate = %0.2f Drawing stopped  @ %0.2f %0.2f -> %0.2f %0.2f %0.2f \n",
                    lastFramerate,
                    posStack[POS_X],
                    posStack[POS_Y],
                    pos[POS_ANGLEX],
                    pos[POS_ANGLEY],
                    pos[POS_ANGLEZ]
              );
    } else
    {
      fprintf(stderr,"Scene not declared..\n");
      return 0;
    }

   glPopMatrix();
  return 1 ;
}
