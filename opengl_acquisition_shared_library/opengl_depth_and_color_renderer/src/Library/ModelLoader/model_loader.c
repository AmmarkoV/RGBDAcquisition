
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <math.h>

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

//For now only using the fixed pipeline renderer..
#include "../Rendering/FixedPipeline/ogl_fixed_pipeline_renderer.h"
#include "../Rendering/ogl_rendering.h"

#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"

#include "model_loader.h"
#include "model_loader_hardcoded.h"
#include "model_loader_obj.h"
#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"
#include "model_editor.h"

#include "../Tools/tools.h"




#define DISABLE_GL_CALL_LIST 0
#if DISABLE_GL_CALL_LIST
 #warning "Please note that glCallList is disabled and that has a really bad effect on graphics card performance"
#endif // DISABLE_GL_CALL_LIST

#define SPHERE_QUALITY 10 /*100 is good quality*/

#define PIE 3.14159265358979323846
#define degreeToRadOLD(deg) (deg)*(PIE/180)



#define USE_QUESTIONMARK_FOR_FAILED_LOADED_MODELS 1


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

const GLfloat defaultAmbient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat defaultDiffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat defaultSpecular[]   = { 0.1f, 0.1f, 0.1f, 1.0f };
const GLfloat defaultShininess[]  = { 5.0f };




int growModelList(struct ModelList * modelStorage,unsigned int framesToAdd)
{/*
  if (framesToAdd == 0) { return 0 ; }
  if (streamObj == 0) { fprintf(stderr,"Given an empty stream to grow \n"); return 0 ; }


  struct KeyFrame * new_frame;
  new_frame = (struct KeyFrame *) realloc( streamObj->frame, sizeof(struct KeyFrame)*( streamObj->MAX_numberOfFrames+framesToAdd ));

   if (new_frame == 0 )
    {
       fprintf(stderr,"Cannot add %u frames to our currently %u sized frame buffer\n",framesToAdd,streamObj->MAX_numberOfFrames);
       return 0;
    } else
     {
      //Clean up all new object types allocated
      void * clear_from_here  =  new_frame+streamObj->MAX_numberOfFrames;
      memset(clear_from_here,0,framesToAdd * sizeof(struct KeyFrame));
    }

   streamObj->MAX_numberOfFrames+=framesToAdd;
   streamObj->frame = new_frame ;*/
  return 1;
}


struct ModelList *  allocateModelList(unsigned int initialSpace)
{
  fprintf(stderr,"allocateModelList(%u)\n",initialSpace);
  struct ModelList * newModelList = (struct ModelList * ) malloc(sizeof(struct ModelList));

  if (newModelList!=0)
  {
    newModelList->currentNumberOfModels = 0;
    newModelList->MAXNumberOfModels = 0;
    newModelList->GUARD_BYTE=123123;


    newModelList->models = (struct Model * ) malloc( initialSpace * sizeof(struct Model));

    if (newModelList->models!=0)
    {
      memset(newModelList->models,0,initialSpace * sizeof(struct Model));
      newModelList->MAXNumberOfModels = initialSpace;
    } else
    {
     fprintf(stderr,RED "ERROR : Failed allocating space for Models to be loaded in \n" NORMAL);
    }
  } else
  {
     fprintf(stderr,RED "ERROR : Failed allocating space for Model List to be loaded in \n" NORMAL);
  }
  return newModelList;
}

int deallocateModelList(struct ModelList* modelStorage)
{
  fprintf(stderr,"Deallocation does not work yet todo : \n");

  if (modelStorage==0) { return 1; }
  if (modelStorage->models!=0)
    {
       unsigned int i=0;

       for (i=0; i<modelStorage->currentNumberOfModels; i++)
       {
           unloadModel(&modelStorage->models[i]);
       }
       free(modelStorage->models);
    }
  free(modelStorage);
  return 0;
}



int printModelList(struct ModelList* modelStorage)
{
  fprintf(stderr,"Model List ___________________________________________\n");
  fprintf(stderr,"%u/%u full ___________________________________________\n" , modelStorage->currentNumberOfModels , modelStorage->MAXNumberOfModels);
  unsigned int i=0;
  for (i=0; i<modelStorage->currentNumberOfModels; i++)
  {
    fprintf(stderr,"%03u | " ,i);
    fprintf(stderr,YELLOW "%s" NORMAL , modelTypeNames[ modelStorage->models[i].type ]);
    fprintf(stderr," %s - %u joints \n" ,
            modelStorage->models[i].pathOfModel ,
            modelStorage->models[i].numberOfBones);
  }
  fprintf(stderr,"______________________________________________________\n");
 return 1;
}

int modelHasASkinTransformation(struct Model * model,float* joints)
{
  if (model==0) {return 0; }
 return  ( (joints!=0) && (model->type==TRI_MODEL) );
}

unsigned int updateModelPosition(struct Model * model,float * position)
{
      if (model==0) { return 0; }
      if (position==0) { return 0; }

      int viewport[4];
      float modelview[16];
      float projection[16];

      //float winX, winY, winZ=0.0;
      float win[3]={0};


      #warning "It is inneficient to query all the tables for each position update..!"
      //fprintf(stderr,"This should not work it should only be called when the draw operation is ready , otherwise the matrices received here are irrelevant\n");
      glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
      glGetFloatv( GL_PROJECTION_MATRIX, projection );
      glGetIntegerv( GL_VIEWPORT, viewport );


    if (model->type==OBJ_MODEL)
     {
      float posX,posY,posZ;
      struct OBJ_Model * modelOBJ = (struct OBJ_Model *) model->modelInternalData;

      posX=position[0] - modelOBJ->boundBox.min.x;
      posY=position[1] - modelOBJ->boundBox.min.y;
      posZ=position[2] - modelOBJ->boundBox.min.z;
      _glhUnProjectf( posX * model->scaleX , posY* model->scaleY , posZ * model->scaleZ , modelview, projection, viewport, win); //gluProject
      model->bbox2D[0]=win[0];
      model->bbox2D[1]=win[1];

      posX=position[0] + modelOBJ->boundBox.max.x;
      posY=position[1] + modelOBJ->boundBox.max.y;
      posZ=position[2] + modelOBJ->boundBox.max.z;
      _glhUnProjectf( posX* model->scaleX , posY* model->scaleY , posZ * model->scaleZ , modelview, projection, viewport, win); //gluProject
      model->bbox2D[2]=win[0];
      model->bbox2D[3]=win[1];

      return 1;
     }

 return 0;
}


unsigned int findModel(struct ModelList * modelStorage ,const char * directory,const char * modelname ,int * found  )
{
  if (modelStorage==0)                    { fprintf(stderr,RED "Error cannot find model in empty modelStorage \n" NORMAL);              return 0; }
  if (modelStorage->models==0)            { fprintf(stderr,RED "Error cannot find allocated space for models..\n" NORMAL);              return 0; }
  if (modelStorage->MAXNumberOfModels==0) { fprintf(stderr,RED "Do not need to search for model in unallocated model list..\n" NORMAL); return 0; }

  if (modelStorage->MAXNumberOfModels <= modelStorage->currentNumberOfModels)
  {
    fprintf(stderr,RED "ERROR : we have loaded more than max possible models ? %u/%u models.. \n" NORMAL,modelStorage->currentNumberOfModels , modelStorage->MAXNumberOfModels);
    return 0;
  }

  fprintf(stderr,"find model called will search among %u models.. \n",modelStorage->currentNumberOfModels);

  *found = 0;
  char tmpPathOfModel[MAX_MODEL_PATHS]={0};
  snprintf(tmpPathOfModel,MAX_MODEL_PATHS,"%s/%s",directory,modelname);

  unsigned int i=0;
  for (i=0; i<modelStorage->currentNumberOfModels; i++)
  {
   if (&modelStorage->models[i]==0)
   {
     fprintf(stderr,RED "Model entry %u is corrupted .. \n" NORMAL,i);
   } else
   if (modelStorage->models[i].pathOfModel!=0)
   {
    if ( strcmp(tmpPathOfModel,modelStorage->models[i].pathOfModel)==0)
    {
      *found=1;
      fprintf(stderr,GREEN "model ( %s ) is already loaded , no need to reload it \n" NORMAL,tmpPathOfModel);
      return i;
    }
   }
  }

return 0;
}



unsigned int loadModel(
                        struct ModelList* modelStorage ,
                        unsigned int whereToLoadModel ,
                        const char * directory,
                        const char * modelname ,
                        const char * extension
                      )
{
  fprintf(stderr,"loadModel ..  \n");
  if ( (directory==0) || (modelname==0) )
  {
    fprintf(stderr,RED "loadModel failing , no modelname given \n" NORMAL );
    return 0;
  }

  if (modelStorage->MAXNumberOfModels <= whereToLoadModel)
  {
    fprintf(stderr,RED "Cannot load model in slot %u/%u \n" NORMAL , whereToLoadModel, modelStorage->MAXNumberOfModels);
    return 0;
  }

  char fullPathToFile[MAX_MODEL_PATHS]={0};
  snprintf(fullPathToFile, MAX_MODEL_PATHS , "%s/%s.%s" , directory , modelname , extension);


  struct Model * mod =  &modelStorage->models[whereToLoadModel];
  if ( mod == 0 )  { fprintf(stderr,"Could not allocate enough space for model %s \n",modelname);  return 0; }
  memset(mod , 0 , sizeof(struct Model));


  //By default any new model will have the old RPY rotation order
  //This can be changed later however..
  //This only has to do with the base rotation, each joint can have a different rotation order..
  mod->rotationOrder = ROTATION_ORDER_RPY;

  snprintf(mod->pathOfModel,MAX_MODEL_PATHS,"%s",fullPathToFile);

  unsigned int unableToLoad=1;
  unsigned int checkForHardcodedReturn=0;
  unsigned int modType = isModelnameAHardcodedModel(modelname,&checkForHardcodedReturn);

  if (checkForHardcodedReturn)
  {
      mod->type = modType;
      mod->modelInternalData = 0;
      unableToLoad=0;
  } else
 if ( strcmp(extension,"tri") == 0 )
    {
      mod->type = TRI_MODEL;
      fprintf(stderr,YELLOW "loading tri model \n" NORMAL);

      struct TRI_Model * triModel = allocateModelTri();
      if (triModel !=0 )
         {
          if ( loadModelTri(fullPathToFile, triModel) )
            {
             mod->numberOfBones = triModel->header.numberOfBones;
             mod->modelInternalData=(void * ) triModel;
             unableToLoad=0;
             //colorCodeBones(triModel); //enable this to color code bones..

/*
             float cylA[3]={0.0,0.0,0.0};
             float cylB[3]={0.0,1.0,0.0};
             punchHoleThroughModel(
                                   triModel ,
                                   cylA ,
                                   cylB ,
                                   0.3 ,
                                   100.0
                                  );
*/

             fprintf(stderr,GREEN " success \n" NORMAL);
            } else
            { fprintf(stderr,RED " unable to load TRI model \n" NORMAL);
              freeModelTri(triModel);
            }
         } else { fprintf(stderr,RED " unable to allocate model \n" NORMAL); }
    } else
 if ( strcmp(extension,"ply") == 0 )
    {
      mod->type = OBJ_ASSIMP_MODEL;
      fprintf(stderr,RED "TODO : loading ply model \n" NORMAL);
    } else
  if ( strcmp(extension,"obj") == 0 )
    {
      mod->type = OBJ_MODEL;
      struct  OBJ_Model *  newObj = (struct  OBJ_Model * ) loadObj(directory,modelname,1);
      mod->modelInternalData = newObj;//(struct  OBJ_Model * ) loadObj(directory,modelname);
      if (mod->modelInternalData !=0 )
         {
             unableToLoad=0;
             //Populate 3D bounding box data
             mod->minX = newObj->minX; mod->minY = newObj->minY;  mod->minZ = newObj->minZ;
             mod->minX = newObj->maxX; mod->maxY = newObj->maxY;  mod->maxZ = newObj->maxZ;
             fprintf(stderr,"new obj : min %0.2f %0.2f %0.2f  max %0.2f %0.2f %0.2f \n",newObj->minX,newObj->minY,newObj->minZ,newObj->maxX,newObj->maxY,newObj->maxZ);
         }
    }


  if (unableToLoad)
    {
      fprintf(stderr,"Could not load object %s \n",modelname);
      fprintf(stderr,"Searched in directory %s \n",directory);
      fprintf(stderr,"Object %s was also not one of the hardcoded shapes\n",modelname);
      if (mod->modelInternalData==0 )
         {
          #if USE_QUESTIONMARK_FOR_FAILED_LOADED_MODELS
              mod->type = OBJ_QUESTION;
              mod->modelInternalData = 0;
              fprintf(stderr,RED "Failed to load object %s , will pretend it got loaded and use a fake object question mark instead\n" NORMAL,modelname);
          #else
            free(mod);
            return 0 ;
          #endif // USE_QUESTIONMARK_FOR_FAILED_LOADED_MODELS
         }
    }

  mod->GUARD_BYTE = modelStorage->GUARD_BYTE;
  mod->initialized=1;
  return 1;
}

void unloadModel(struct Model * mod)
{
   if (mod == 0 ) { return ; }

    switch ( mod->type )
    {
      case TRI_MODEL :
          freeModelTri( (struct TRI_Model *) mod->modelInternalData);
      break;
      case OBJ_MODEL :
          unloadObj( (struct  OBJ_Model * ) mod->modelInternalData);
      break;
    };
}


int loadModelToModelList(struct ModelList* modelStorage,const char * modelDirectory,const char * modelName , const char * modelExtension , unsigned int * whereModelWasLoaded)
{
  if (modelStorage==0) { fprintf(stderr,"cannot loadModelToModelList without an allocated model list..\n"); return 0; }
  fprintf(stderr,"loadModelToModelList called (dir %s , name %s , ext %s )  , %u/%u .. \n",modelDirectory,modelName,modelExtension , modelStorage->currentNumberOfModels , modelStorage->MAXNumberOfModels );

  int foundAlreadyExistingModel=0;
  unsigned int modelLocation = findModel(modelStorage,modelDirectory,modelName, &foundAlreadyExistingModel);
 // fprintf(stderr,"findModel survived .. \n");

  if (!foundAlreadyExistingModel)
   { //If we can't find an already loaded version of the mesh we are looking for
     unsigned int whereToLoadModel=modelStorage->currentNumberOfModels;
     fprintf(stderr,"before LoadModel @ %u .. \n",whereToLoadModel);
     if (loadModel(modelStorage,whereToLoadModel,modelDirectory,modelName,modelExtension))
      {
        fprintf(stderr,GREEN "Model %s is now loaded as model[%u] \n" NORMAL,modelName , whereToLoadModel );
        *whereModelWasLoaded=whereToLoadModel;
        modelStorage->currentNumberOfModels+=1;
        return 1;
      } else
      { fprintf(stderr,RED "Failed loading new model %s ( %u ) \n" NORMAL,modelName, whereToLoadModel );        return 0; }
    } else
    {
     *whereModelWasLoaded=modelLocation;
     fprintf(stderr,GREEN "Model %s found already loaded @ %u \n" NORMAL,modelName,modelLocation);
     return 1;
    }
 return 0;
}



int drawTRIModel(struct VirtualStream * scene , struct Model * mod)
{
  struct TRI_Model * tri = (struct TRI_Model *) mod->modelInternalData;

  if (mod->showSkeleton)
          {
           /* Joints Drawing */
           unsigned int outputNumberOfJoints;
           unsigned int * parentNode= convertTRIBonesToParentList( tri , &outputNumberOfJoints); // outputNumberOfJoints will be overwritten
           float * jointPositions = convertTRIBonesToJointPositions( tri , &outputNumberOfJoints );
           if (jointPositions!=0)
             {
              renderOGLBones(jointPositions,parentNode,outputNumberOfJoints);
              free(jointPositions);
             }
          } else
          {
           //doOGLGenericDrawCalllist
             renderOGL
            (  //TODO
                                    0,// scene->activeProjectionMatrix,
                                    0,// scene->activeViewMatrix,
                                    0,// scene->activeModelMatrix,
                                    0,// scene->activeModelViewProjectionMatrix,
                                     tri->vertices ,       tri->header.numberOfVertices ,
                                     tri->normal ,         tri->header.numberOfNormals ,
                                     tri->textureCoords ,  tri->header.numberOfTextureCoords ,
                                     tri->colors ,         tri->header.numberOfColors ,
                                     tri->indices ,        tri->header.numberOfIndices
                             );

          }
  return 1;
}




int drawOBJModel(struct VirtualStream * scene ,struct Model * mod)
{
 //fprintf(stderr,"drawing OBJ model\n");
 if (mod->modelInternalData!=0)
         {
           if (mod->highlight)
           {
            struct  OBJ_Model *  drawOBJ = (struct  OBJ_Model * ) mod->modelInternalData;
            drawBoundingBox(0,0,0,drawOBJ->minX,drawOBJ->minY,drawOBJ->minZ,drawOBJ->maxX,drawOBJ->maxY,drawOBJ->maxZ);
           }

           //A model has been created , and it can be served
           GLuint objlist  =  getObjOGLList( ( struct OBJ_Model * ) mod->modelInternalData);     checkOpenGLError(__FILE__, __LINE__);

           if ( (objlist!=0) && (!DISABLE_GL_CALL_LIST) )
             { //We have compiled a list of the triangles for better performance
               glCallList(objlist);                                                              checkOpenGLError(__FILE__, __LINE__);
             }  else
             { //Just feed the triangles to open gl one by one ( slow )
               drawOBJMesh( ( struct OBJ_Model * ) mod->modelInternalData);                      checkOpenGLError(__FILE__, __LINE__);
             }
         } else
         { fprintf(stderr,"Could not draw unspecified model\n"); }

 glDisable(GL_TEXTURE_2D);
 //TODO : <-- change drawOBJMesh , Calllist so that they dont leave textures on! :P
 return 1;
}





int drawModelAt(
                 struct Model * mod,
                 float positionX,
                 float positionY,
                 float positionZ,
                 float rotationX,//heading,
                 float rotationY,//pitch,
                 float rotationZ,//roll,
                 unsigned int rotationOrder
                )
{
 if (mod==0)
  {
    fprintf(stderr,"Cannot draw model at position %0.2f %0.2f %0.2f , it doesnt exist \n",positionX,positionY,positionZ);
    return 0;
  }

  if (checkOpenGLError(__FILE__, __LINE__))
  {
     fprintf(stderr,"drawModelAt called while on an erroneous state :(\n");
  }


  glPushMatrix();
  //glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

   if (mod->wireframe)  { glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ); } else
                        { glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ); }
  //glEnable(GL_NORMALIZE);
  /*If scale factors other than 1 are applied to the modelview matrix
    and lighting is enabled, lighting often appears wrong.
    In that case, enable automatic normalization of normals by
    calling glEnable with the argument GL_NORMALIZE.*/

  if (mod->nocull) { glDisable(GL_CULL_FACE); }


   //fprintf(stderr,"drawModelAt: %s -> %u \n", mod->pathOfModel , rotationOrder);
   struct Matrix4x4OfFloats modelTransformation={0};
   create4x4FModelTransformation(
                                  &modelTransformation,
                                  //Rotation Component
                                  (float) rotationX,//heading,
                                  (float) rotationY,//pitch,
                                  (float) rotationZ,//roll,
                                           rotationOrder,
                                  //Translation Component
                                  (float) positionX,
                                  (float) positionY,
                                  (float) positionZ ,
                                  //Scale Component
                                  (float) mod->scaleX,
                                  (float) mod->scaleY,
                                  (float) mod->scaleZ
                                 );
  transpose4x4FMatrix(modelTransformation.m); //Because we want to use this in OpenGL
  glMultMatrixf(modelTransformation.m);


/*
  SEE Matrices..*/
  int viewport[4];
  float modelview[16];
  float projection[16];
  glGetFloatv( GL_MODELVIEW_MATRIX, modelview );
  glGetFloatv( GL_PROJECTION_MATRIX, projection );
  glGetIntegerv( GL_VIEWPORT, viewport );
  print4x4FMatrix("Projection",projection,0);
  print4x4FMatrix("ModelView",modelview,0);
  print4x4FMatrix("ModelTransform",modelTransformation.m,0);
 // exit (0);

  if (checkOpenGLError(__FILE__, __LINE__)) { fprintf(stderr,"drawModelAt error after specifying dimensions \n"); }

       // MAGIC NO COLOR VALUE :P MEANS NO COLOR SELECTION
      if (mod->nocolor!=0)  { glDisable(GL_COLOR_MATERIAL);   } else
      {
        //We Have a color to set
        glEnable(GL_COLOR_MATERIAL);                            checkOpenGLError(__FILE__, __LINE__);


        GLenum faces=GL_FRONT;//GL_FRONT_AND_BACK;
        glMaterialfv(faces, GL_AMBIENT,    defaultAmbient);     checkOpenGLError(__FILE__, __LINE__);
        glMaterialfv(faces, GL_DIFFUSE,    defaultDiffuse);     checkOpenGLError(__FILE__, __LINE__);
        glMaterialfv(faces, GL_SPECULAR,   defaultSpecular);    checkOpenGLError(__FILE__, __LINE__);
        glMaterialfv(faces, GL_SHININESS,   defaultShininess);  checkOpenGLError(__FILE__, __LINE__);


        if (mod->transparency==0.0)
        {
          glColor3f(mod->colorR,mod->colorG,mod->colorB);       checkOpenGLError(__FILE__, __LINE__);
        }
        else
        {
          // Turn Blending On
          glEnable(GL_BLEND);	                                             checkOpenGLError(__FILE__, __LINE__);
          glBlendFunc(GL_SRC_ALPHA, GL_ONE);                                 checkOpenGLError(__FILE__, __LINE__);
          glColor4f(mod->colorR,mod->colorG,mod->colorB,mod->transparency);  checkOpenGLError(__FILE__, __LINE__);
        }
      }

  //---------------------------------------------------------------------------
  if (checkOpenGLError(__FILE__, __LINE__))
    { fprintf(stderr,"drawModelAt error after specifying color/materials\n"); }
  //---------------------------------------------------------------------------


  //We have all the files
  float position[7]={0};
  updateModelPosition(mod,position);

    switch(mod->type)
    {                                  //TODO: drawTRIModel(0,mod);
        case TRI_MODEL        :  drawTRIModel(0,mod); break;
        case OBJ_MODEL        :  drawOBJModel(0,mod); break;
        case OBJ_ASSIMP_MODEL :  fprintf(stderr,"TODO : drawAssimpModel\n");  break;
        //-----------------------------------------------------------------------------------
        default :
                     if (! drawHardcodedModel(mod->type) )
                        {
                         fprintf(stderr, "Cannot draw model , unknown type %d\n",mod->type );
                        }
        break;
    }

  //---------------------------------------------------------------------------
  if (checkOpenGLError(__FILE__, __LINE__))
     { fprintf(stderr,"drawModelAt error after drawing geometry\n"); }
  //---------------------------------------------------------------------------

  if (mod->transparency!=0.0) {glDisable(GL_BLEND);  }
  if (mod->nocolor)           {glEnable(GL_COLOR); glEnable(GL_COLOR_MATERIAL); }
  if (mod->nocull)            {glEnable(GL_CULL_FACE); }

  //glDisable(GL_NORMALIZE);
  glPopMatrix();
 return 1;
}

int drawModel(struct Model * mod)
{
    if (mod == 0) { fprintf(stderr,"Cannot draw model , it doesnt exist \n"); return 0; } //If mod = 0 accesing the fields below will lead in crashing..
    return drawModelAt(
                        mod,
                        mod->x,
                        mod->y,
                        mod->z,
                        mod->rotationX, //heading
                        mod->rotationY, //pitch
                        mod->rotationZ, //roll
                        mod->rotationOrder
                       );
}


void printBasicModelState(struct Model * mod)
{
  fprintf(stderr,"Model(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f (rotOrder %u))\n",mod->x,mod->y,mod->z,mod->rotationX,mod->rotationY,mod->rotationZ,mod->rotationOrder);
}


int addToModelCoordinates(
                            struct Model * mod,
                            float x,
                            float y,
                            float z,
                            float rotationX, //heading,
                            float rotationY, //pitch,
                            float rotationZ  //roll
                         )
{
  if (mod==0) { return 0; }
  mod->x+=x; mod->y+=y; mod->z+=z;

  mod->rotationX+=rotationX;  //heading
  mod->rotationY+=rotationY;  //pitch
  mod->rotationZ+=rotationZ;  //roll

  printBasicModelState(mod);
  return 1;
}

int addToModelCoordinatesNoSTACK(
                                  struct Model * mod,
                                  float *x,
                                  float *y,
                                  float *z,
                                  float *rotationX,
                                  float *rotationY,
                                  float *rotationZ
                                )
{
  if (mod==0) { return 0; }
  mod->x+=*x; mod->y+=*y; mod->z+=*z;

  mod->rotationX+=*rotationX;  //heading
  mod->rotationY+=*rotationY;  //pitch
  mod->rotationZ+=*rotationZ;  //roll

  printBasicModelState(mod);
  return 1;
}

int setModelCoordinates(
                         struct Model * mod,
                         float x,
                         float y,
                         float z,
                         float rotationX, //heading,
                         float rotationY, //pitch,
                         float rotationZ  //roll
                       )
{
  if (mod==0) { return 0; }
  fprintf(stderr,"Model SET Got params(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",x,y,z,rotationX,rotationY,rotationZ);

  mod->x=x; mod->y=y; mod->z=z;

  mod->rotationX=rotationX; mod->rotationY=rotationY; mod->rotationZ=rotationZ;
  fprintf(stderr,"Model SET (%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",mod->x,mod->y,mod->z,mod->rotationX,mod->rotationY,mod->rotationZ);
  return 1;
}

int setModelCoordinatesNoSTACK(
                                  struct Model * mod,
                                  float *x,
                                  float *y,
                                  float *z,
                                  float *rotationX,
                                  float *rotationY,
                                  float *rotationZ
                              )
{
  if (mod==0) { return 0; }
  //fprintf(stderr,"Model SET NoSTACK Got params(%0.2f %0.2f %0.2f - %0.4f %0.4f %0.4f)\n",*x,*y,*z,*heading,*pitch,*roll);

  mod->x=*x; mod->y=*y; mod->z=*z;

  mod->rotationX=*rotationX; mod->rotationY=*rotationY; mod->rotationZ=*rotationZ;
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
 struct OBJ_Model * objMod = (struct OBJ_Model * ) mod->modelInternalData;
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


int getModelListBoneNumber(struct ModelList * modelStorage,unsigned int modelNumber)
{
  fprintf(stderr,"getModelListBoneNumber(modelStorage,modelNumber=%u)\n",modelNumber);
  if (modelNumber>=modelStorage->currentNumberOfModels)
  {
    fprintf(stderr," modelNumber is out of bounds\n");
    return 0;
  }

  return modelStorage->models[modelNumber].numberOfBones;
}

int getModelBoneIDFromBoneName(struct Model *mod,const char * boneName,int * found)
{
//fprintf(stderr,"getModelBoneIDFromBoneName(boneName=%s)\n",boneName);
if (found==0) { return 0; }
 *found=0;

if (mod==0)   { return 0; }
if (boneName==0)   { return 0; }

if (mod->initialized!=1)
{
  fprintf(stderr,"model is not initialized not doing getModelBoneIDFromBoneName(boneName=%s)\n",boneName);
  return 0;
}
 //fprintf(stderr,"Searching model %s for a bone named %s \n",mod->pathOfModel , boneName);

 if (mod->type==TRI_MODEL)
 {
  struct TRI_Model * triM = (struct TRI_Model * ) mod->modelInternalData;
  if (triM!=0)
   {
     unsigned int i=0;
     unsigned int numberOfBones=mod->numberOfBones;
    //fprintf(stderr,"getModelBoneIDFromBoneName will search through %u bones \n",numberOfBones);

     for (i=0; i<numberOfBones; i++)
     {
       //fprintf(stderr,"comp %u \n" , i);
       if (strcmp( triM->bones[i].boneName , boneName) == 0 )
       {
        // fprintf(stderr,"found it , it is joint # %u \n" , i);
         *found=1;
         return i;
       }
     }
   }
 } else
 {
  fprintf(stderr,"getModelBoneIDFromBoneName: Unsupported model type\n");
 }

  fprintf(stderr,RED "Searching model %s for a bone named %s , could not find it\n" NORMAL,mod->pathOfModel , boneName);
 return 0;
}



int getModelBoneRotationOrderFromBoneName(struct Model *mod,unsigned int boneID)
{
 if (mod==0)   { return 0; }
 if (mod->initialized!=1)
 {
  fprintf(stderr,"model is not initialized not doing getModelBoneRotationOrderFromBoneName(boneID=%u)\n",boneID);
  return 0;
 }

 if (mod->type==TRI_MODEL)
 {
  struct TRI_Model * triM = (struct TRI_Model * ) mod->modelInternalData;
  if ((triM!=0) && (triM->bones!=0) )
   {
     if (boneID<triM->header.numberOfBones)
     {
        return (int)  triM->bones[boneID].info->eulerRotationOrder;
     }
   }
 } else
 {
  fprintf(stderr,"getModelBoneRotationOrderFromBoneName: Unsupported model type\n");
 }

 return 0;
}
