/* TrajectoryParser..
   A small generic library for keeping an array of 3d Objects their positions and orientations
   moving through time and interpolating/extrapolating them for generating sequences of synthetic data
   typically rendered using OpenGL or something else!

   GITHUB Repo : https://github.com/AmmarkoV/RGBDAcquisition/blob/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/TrajectoryParser.cpp
   my URLs: http://ammar.gr
   Written by Ammar Qammaz a.k.a. AmmarkoV 2013

   The idea here is create a struct VirtualObject * pointer by issuing
   readVirtualStream(struct VirtualStream * newstream , char * filename); or  createVirtualStream(char * filename);
   and then using a file that contains objects and their virtual coordinates , or calls like
   addObjectToVirtualStream
    or
   addPositionToObject
   populate the Virtual stream with objects and their positions

   after that we can query the positions of the objects using calculateVirtualStreamPos orcalculateVirtualStreamPosAfterTime and get back our object position
   for an arbitrary moment of our world

   After finishing with the VirtualObject stream  it should be destroyed using destroyVirtualStream in order for the memory to be gracefully freed
*/


#include "TrajectoryParser.h"
#include "TrajectoryParserDataStructures.h"
#include "TrajectoryCalculator.h"
#include "TrajectoryPrimitives.h"

#include "../OGLRendererSandbox.h"
#include "../ModelLoader/model_loader.h"
#include "../../../../../tools/AmMatrix/matrixCalculations.h"
#include "../../../../../tools/AmMatrix/quaternions.h"
#include "../Tools/tools.h"

//Using normalizeQuaternionsTJP #include "../../../../tools/AmMatrix/matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


//If you want Trajectory parser to be able to READ
//and parse files you should set  USE_FILE_INPUT  to 1
#define USE_FILE_INPUT 1


#if USE_FILE_INPUT
  #include "InputParser_C.h"
#endif


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


float depthMemoryOutputScale=0.1;

//int (*saveSnapshot) (int,struct calibration *);
int doneAddingObjectsToVirtualStream(struct VirtualStream * newstream)
{
               hashMap_PrepareForQueries(newstream->objectTypesHash);
               hashMap_PrepareForQueries(newstream->objectHash);
               hashMap_PrepareForQueries(newstream->connectorHash);
               hashMap_PrepareForQueries(newstream->eventHash);


               /*
               hashMap_Print(newstream->objectTypesHash,"Object Types");
               hashMap_Print(newstream->objectHash,"Object");
               hashMap_Print(newstream->connectorHash,"Connectors");
               hashMap_Print(newstream->eventHash,"Events");
               exit(0);
               */
    return 1;
}

int writeVirtualStream(struct VirtualStream * newstream,const char * filename)
{
  if (newstream==0) { fprintf(stderr,"Cannot writeVirtualStream(%s) , virtual stream does not exist\n",filename); return 0; }
  FILE * fp = fopen(filename,"w");
  if (fp == 0 ) { fprintf(stderr,"Cannot open trajectory stream output file %s \n",filename); return 0; }

  fprintf(fp,"#Automatically generated virtualStream(initial file was %s)\n\n",newstream->filename);
  fprintf(fp,"#Some generic settings\n");
  fprintf(fp,"AUTOREFRESH(%u)\n",newstream->autoRefresh);

  if (newstream->ignoreTime) { fprintf(fp,"INTERPOLATE_TIME(0)\n"); } else
                             { fprintf(fp,"INTERPOLATE_TIME(1)\n"); }
  fprintf(fp,"\n\n#List of object-types\n");

  unsigned int i=0;
  for (i=1; i<newstream->numberOfObjectTypes; i++) { fprintf(fp,"OBJECTTYPE(%s,\"%s\")\n",newstream->objectTypes[i].name,newstream->objectTypes[i].model); }


  fprintf(fp,"\n\n#List of objects and their positions");
  unsigned int pos=0;
  for (i=1; i<newstream->numberOfObjects; i++)
    {
      fprintf(fp,"\nOBJECT(%s,%s,%u,%u,%u,%u,%u,%0.2f,%0.2f,%0.2f,%s)\n",
              newstream->object[i].name,
              newstream->object[i].typeStr,
              (unsigned int) newstream->object[i].R/255,
              (unsigned int) newstream->object[i].G/255,
              (unsigned int) newstream->object[i].B/255,
              (unsigned int) newstream->object[i].Transparency/255,
              newstream->object[i].nocolor,
              newstream->object[i].scaleX,
              newstream->object[i].scaleY,
              newstream->object[i].scaleZ,
              newstream->object[i].value);

      for (pos=0; pos < newstream->object[i].numberOfFrames; pos++)
      {
         fprintf(fp,"POS(%s,%u,%f,%f,%f,%f,%f,%f,%f)\n",
                     newstream->object[i].name ,
                     newstream->object[i].frame[pos].time,

                     newstream->object[i].frame[pos].x,
                     newstream->object[i].frame[pos].y,
                     newstream->object[i].frame[pos].z,

                     newstream->object[i].frame[pos].rot1,
                     newstream->object[i].frame[pos].rot2,
                     newstream->object[i].frame[pos].rot3,
                     newstream->object[i].frame[pos].rot4
                );
      }
    }

  fclose(fp);
  return 1;
}


int processCommand( struct VirtualStream * newstream , struct ModelList * modelStorage, struct InputParserC * ipc , unsigned int label , char * line , unsigned int words_count )
{

  char name[MAX_PATH]={0};
  char nameB[MAX_PATH]={0};
  char model[MAX_PATH]={0};
  char typeStr[MAX_PATH]={0};
  char includeFile[MAX_PATH]={0};
  float euler[3];
  float quaternions[4];
  float pos[16]={0};
  unsigned int i,satteliteObj,planetObj,item,frame,duration,time,coordLength,eventType=0,foundA=0,foundB=0,objIDA=0,objIDB=0;


  if (line[0]=='#')
     { label = TRAJECTORYPRIMITIVES_COMMENT; }

  if (newstream->debug)
     {
      fprintf(stderr,"Label %u =>  Line  %s / words %u \n",label,line,words_count);
     }


  //Initialize this so renderer does not explode..!
  newstream->controls.lastTickMillisecond = GetTickCountMilliseconds();

  switch (label)
  {
             case TRAJECTORYPRIMITIVES_COMMENT : /*Comment , don't spam console etc*/ break;
             case TRAJECTORYPRIMITIVES_OBJ : break;
             case TRAJECTORYPRIMITIVES_ARROW  : break;
             case TRAJECTORYPRIMITIVES_SILENT : newstream->silent=1;  break;
             case TRAJECTORYPRIMITIVES_DEBUG                             :  newstream->debug=1;           break;

             case TRAJECTORYPRIMITIVES_GENERATE_ANGLE_OBJECTS            :

                 InputParser_GetWord(ipc,1,model,MAX_PATH);
                 generateAngleObjectsForVirtualStream(newstream,modelStorage,model);
               break;


             case TRAJECTORYPRIMITIVES_NEAR_CLIP                         :  newstream->controls.nearPlane=InputParser_GetWordFloat(ipc,1); break;
             case TRAJECTORYPRIMITIVES_FAR_CLIP                          :  newstream->controls.farPlane=InputParser_GetWordFloat(ipc,1); break;
             case TRAJECTORYPRIMITIVES_TIMESTAMP                         :  newstream->timestamp=InputParser_GetWordInt(ipc,1); break;
             case TRAJECTORYPRIMITIVES_AUTOREFRESH                       :  newstream->autoRefresh = InputParser_GetWordInt(ipc,1); break;
             case TRAJECTORYPRIMITIVES_INTERPOLATE_TIME                  :  newstream->ignoreTime = ( InputParser_GetWordInt(ipc,1) == 0 ); break;
             case TRAJECTORYPRIMITIVES_ALWAYS_SHOW_LAST_FRAME            :  newstream->alwaysShowLastFrame = InputParser_GetWordInt(ipc,1)  ; break;

             case TRAJECTORYPRIMITIVES_FRAME_RESET                       :   newstream->timestamp=0;     break;
             case TRAJECTORYPRIMITIVES_FRAME                             :   newstream->timestamp+=100;  break;
             case TRAJECTORYPRIMITIVES_RATE                              :   newstream->rate=InputParser_GetWordFloat(ipc,1);  newstream->forceRateRegardlessOfGPUSpeed=1; break;
             case TRAJECTORYPRIMITIVES_MOVE_VIEW                         :   newstream->userCanMoveCameraOnHisOwn=InputParser_GetWordInt(ipc,1); break;
             case TRAJECTORYPRIMITIVES_SMOOTH                            :   smoothTrajectories(newstream); break;
             case TRAJECTORYPRIMITIVES_OBJ_OFFSET                        :   newstream->objDeclarationsOffset = InputParser_GetWordInt(ipc,1);   break;
             case TRAJECTORYPRIMITIVES_HAND_POINTS                       : break;

             case TRAJECTORYPRIMITIVES_BACKGROUND  :
                newstream->backgroundR = (float) InputParser_GetWordInt(ipc,1) / 255;
                newstream->backgroundG = (float) InputParser_GetWordInt(ipc,2) / 255;
                newstream->backgroundB = (float) InputParser_GetWordInt(ipc,3) / 255;
             break;


             case TRAJECTORYPRIMITIVES_AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES  :
               satteliteObj = 1 + newstream->objDeclarationsOffset + InputParser_GetWordInt(ipc,1);    /*Item 0 is camera so we +1 */
               planetObj    = 1 + newstream->objDeclarationsOffset + InputParser_GetWordInt(ipc,2);    /*Item 0 is camera so we +1 */
               frame     = InputParser_GetWordInt(ipc,3);
               duration  = InputParser_GetWordInt(ipc,4);
               if (! affixSatteliteToPlanetFromFrameForLength(newstream,satteliteObj,planetObj,frame,duration) )
               {
                fprintf(stderr,RED "Could not affix Object %u to Object %u for %u frames ( starting @ %u )\n" NORMAL , satteliteObj,planetObj,duration,frame);
               }
             break;


             case TRAJECTORYPRIMITIVES_SHADER :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,nameB,MAX_PATH);

               enableShaders(name,nameB);
             break;





             case TRAJECTORYPRIMITIVES_INCLUDE :
              InputParser_GetWord(ipc,1,includeFile,MAX_PATH);
              fprintf(stderr,YELLOW "Including.. %s..!\n" NORMAL,includeFile);
              if (appendVirtualStreamFromFile(newstream,modelStorage,includeFile))
              {
                fprintf(stderr,GREEN "Successfully included file %s..!" NORMAL,includeFile);
              } else
              {
                fprintf(stderr,RED "Could not include file..!" NORMAL);
              }
             break;

            case TRAJECTORYPRIMITIVES_OBJECT_TYPE :
            case TRAJECTORYPRIMITIVES_OBJECTTYPE :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,model,MAX_PATH);
               InputParser_GetWord(ipc,3,nameB,MAX_PATH);

               addObjectTypeToVirtualStream( newstream , name, model , nameB /*Path*/ );
             break;


           case TRAJECTORYPRIMITIVES_SAVED_FILE_DEPTH_SCALE :
                 fprintf(stderr,"Setting output depth scale..!\n");
                 depthMemoryOutputScale = InputParser_GetWordFloat(ipc,1);
           break;

           case TRAJECTORYPRIMITIVES_PROJECTION_MATRIX :
                 newstream->projectionMatrixDeclared=1;
                 for (i=1; i<=16; i++) { newstream->projectionMatrix[i-1] = (float)  InputParser_GetWordFloat(ipc,i); }
                 fprintf(stderr,"Projection Matrix given to TrajectoryParser\n");
           break;


           case TRAJECTORYPRIMITIVES_EMULATE_PROJECTION_MATRIX :
                 newstream->emulateProjectionMatrixDeclared=1;
                 for (i=1; i<=9; i++) { newstream->emulateProjectionMatrix[i-1] = (float)  InputParser_GetWordFloat(ipc,i); }
                 fprintf(stderr,"Emulating Projection Matrix given to TrajectoryParser\n");
           break;


           case TRAJECTORYPRIMITIVES_MODELVIEW_MATRIX :
                 newstream->modelViewMatrixDeclared=1;
                 for (i=1; i<=16; i++) { newstream->modelViewMatrix[i-1] = (float) InputParser_GetWordFloat(ipc,i); }
                 fprintf(stderr,"ModelView Matrix given to TrajectoryParser\n");
           break;


           case TRAJECTORYPRIMITIVES_SCALE_WORLD :
               newstream->scaleWorld[0] = InputParser_GetWordFloat(ipc,1);
               newstream->scaleWorld[1] = InputParser_GetWordFloat(ipc,2);
               newstream->scaleWorld[2] = InputParser_GetWordFloat(ipc,3);
               fprintf(stderr,"Scaling everything * %f %f %f \n",newstream->scaleWorld[0],newstream->scaleWorld[1],newstream->scaleWorld[2]);
           break;


           case TRAJECTORYPRIMITIVES_OFFSET_ROTATIONS :
               newstream->rotationsOffset[0] = InputParser_GetWordFloat(ipc,1);
               newstream->rotationsOffset[1] = InputParser_GetWordFloat(ipc,2);
               newstream->rotationsOffset[2] = InputParser_GetWordFloat(ipc,3);
               fprintf(stderr,"Offsetting rotations + %f %f %f \n",newstream->rotationsOffset[0],newstream->rotationsOffset[1],newstream->rotationsOffset[2]);
           break;



           case TRAJECTORYPRIMITIVES_MAP_ROTATIONS :
               newstream->scaleWorld[3] = InputParser_GetWordFloat(ipc,1);
               newstream->scaleWorld[4] = InputParser_GetWordFloat(ipc,2);
               newstream->scaleWorld[5] = InputParser_GetWordFloat(ipc,3);

               if (InputParser_GetWordChar(ipc,4,0)=='x') { newstream->rotationsXYZ[0]=0; } else
               if (InputParser_GetWordChar(ipc,4,0)=='y') { newstream->rotationsXYZ[0]=1; } else
               if (InputParser_GetWordChar(ipc,4,0)=='z') { newstream->rotationsXYZ[0]=2; }
                //--------------------
               if (InputParser_GetWordChar(ipc,4,1)=='x') { newstream->rotationsXYZ[1]=0; } else
               if (InputParser_GetWordChar(ipc,4,1)=='y') { newstream->rotationsXYZ[1]=1; } else
               if (InputParser_GetWordChar(ipc,4,1)=='z') { newstream->rotationsXYZ[1]=2; }
                //--------------------
               if (InputParser_GetWordChar(ipc,4,2)=='x') { newstream->rotationsXYZ[2]=0; } else
               if (InputParser_GetWordChar(ipc,4,2)=='y') { newstream->rotationsXYZ[2]=1; } else
               if (InputParser_GetWordChar(ipc,4,2)=='z') { newstream->rotationsXYZ[2]=2; }

               newstream->rotationsOverride=1;

               fprintf(stderr,"Mapping rotations to  %f %f %f / %u %u %u \n",
                       newstream->scaleWorld[3] , newstream->scaleWorld[4] ,newstream->scaleWorld[5] ,
                         newstream->rotationsXYZ[0],newstream->rotationsXYZ[1],newstream->rotationsXYZ[2]);
           break;


           case TRAJECTORYPRIMITIVES_RIGID_OBJECT :
           case TRAJECTORYPRIMITIVES_OBJECT :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,typeStr,MAX_PATH);
               addObjectToVirtualStream(newstream ,
                                        modelStorage, name,typeStr,
                                        (unsigned char) InputParser_GetWordInt(ipc,3),
                                        (unsigned char) InputParser_GetWordInt(ipc,4),
                                        (unsigned char) InputParser_GetWordInt(ipc,5),
                                        (unsigned char) InputParser_GetWordInt(ipc,6),
                                        (unsigned char) InputParser_GetWordInt(ipc,7),
                                        0,0,
                                        InputParser_GetWordFloat(ipc,8),
                                        InputParser_GetWordFloat(ipc,9),
                                        InputParser_GetWordFloat(ipc,10)
                                        ,0);

          break;



           case TRAJECTORYPRIMITIVES_COMPOSITE_OBJECT :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,typeStr,MAX_PATH);
               addObjectToVirtualStream(newstream ,
                                        modelStorage, name,typeStr,
                                        (unsigned char) InputParser_GetWordInt(ipc,3),
                                        (unsigned char) InputParser_GetWordInt(ipc,4),
                                        (unsigned char) InputParser_GetWordInt(ipc,5),
                                        (unsigned char) InputParser_GetWordInt(ipc,6),
                                        (unsigned char) InputParser_GetWordInt(ipc,7),
                                        0,0,
                                        InputParser_GetWordFloat(ipc,8),
                                        InputParser_GetWordFloat(ipc,9),
                                        InputParser_GetWordFloat(ipc,10),
                                        InputParser_GetWordInt(ipc,11) );
          break;



          case TRAJECTORYPRIMITIVES_CONNECTOR :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,nameB,MAX_PATH);

               addConnectorToVirtualStream(
                                            newstream ,
                                            name , nameB,
                                            (unsigned char) InputParser_GetWordInt(ipc,3),
                                            (unsigned char) InputParser_GetWordInt(ipc,4),
                                            (unsigned char) InputParser_GetWordInt(ipc,5),
                                            (unsigned char) InputParser_GetWordInt(ipc,6),
                                            (float) InputParser_GetWordFloat(ipc,7),
                                            typeStr
                                          );
          break;


          case TRAJECTORYPRIMITIVES_POSE4X4 :
                InputParser_GetWord(ipc,1,name,MAX_PATH);
                time = InputParser_GetWordInt(ipc,2);
                InputParser_GetWord(ipc,3,nameB,MAX_PATH);

                for (i=0; i<16; i++)
                     {
                      pos[i] = InputParser_GetWordFloat(ipc,4+i);
                     }
                coordLength=16;

                addPoseToObjectState( newstream , modelStorage , name  , nameB , time , (float*) pos , coordLength );
          break;


          case TRAJECTORYPRIMITIVES_POSEQ :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               time = InputParser_GetWordInt(ipc,2);
               InputParser_GetWord(ipc,3,nameB,MAX_PATH);

               pos[0] = InputParser_GetWordFloat(ipc,4);
               pos[1] = InputParser_GetWordFloat(ipc,5);
               pos[2] = InputParser_GetWordFloat(ipc,6);
               pos[3] = InputParser_GetWordFloat(ipc,7);
               coordLength=4;

               addPoseToObjectState( newstream , modelStorage , name  , nameB , time , (float*) pos , coordLength );
          break;


          case TRAJECTORYPRIMITIVES_POSE :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               time = InputParser_GetWordInt(ipc,2);
               InputParser_GetWord(ipc,3,nameB,MAX_PATH);

               pos[0] = InputParser_GetWordFloat(ipc,4);
               pos[1] = InputParser_GetWordFloat(ipc,5);
               pos[2] = InputParser_GetWordFloat(ipc,6);
               pos[3] = 0.0;
               coordLength=3;

               //if (newstream->rotationsOverride)
               //      { flipRotationAxis(&pos[3],&pos[4],&pos[5], newstream->rotationsXYZ[0] , newstream->rotationsXYZ[1] , newstream->rotationsXYZ[2]); }
               addPoseToObjectState( newstream , modelStorage , name  , nameB , time , (float*) pos , coordLength );
          break;


          case TRAJECTORYPRIMITIVES_POSE_ROTATION_ORDER :
               //fprintf(stderr,"TRAJECTORYPRIMITIVES_POSE_ROTATION_ORDER recvd\n");
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,nameB,MAX_PATH);
               InputParser_GetLowercaseWord(ipc,3,typeStr,MAX_PATH);

               changeModelJointRotationOrder(
                                             newstream ,
                                             modelStorage,
                                             name  ,
                                             nameB,
                                             typeStr
                                            );
                //fprintf(stderr,"survived\n");
          break;


          case TRAJECTORYPRIMITIVES_OBJECT_ROTATION_ORDER:
               //fprintf(stderr,"TRAJECTORYPRIMITIVES_OBJECT_ROTATION_ORDER recvd\n");
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetLowercaseWord(ipc,2,typeStr,MAX_PATH);

               changeObjectRotationOrder(
                                         newstream ,
                                         modelStorage,
                                         name  ,
                                         typeStr
                                        );
                 //fprintf(stderr,"survived\n");
          break;

          case TRAJECTORYPRIMITIVES_MOVE :
          case TRAJECTORYPRIMITIVES_POS :
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               time = InputParser_GetWordInt(ipc,2);

               pos[0] = newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,3); //X POS
               pos[1] = newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,4); //Y POS
               pos[2] = newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,5); //Z POS
               pos[3] = InputParser_GetWordFloat(ipc,6);
               pos[4] = InputParser_GetWordFloat(ipc,7);
               pos[5] = InputParser_GetWordFloat(ipc,8);
               pos[6] = InputParser_GetWordFloat(ipc,9);
               if ( (pos[3]==0) && (pos[4]==0)  && (pos[5]==0)  && (pos[6]==0)  )
                  {
                    /*fprintf(stderr,"OBJ %u , frame %u declared with completely zero quaternion normalizing it to 0,0,0,1\n",item,newstream->timestamp);*/
                    pos[6]=1.0;
                  }

               /*
               coordLength=7;

               if (newstream->rotationsOverride)
                     { flipRotationAxis(&pos[3],&pos[4],&pos[5], newstream->rotationsXYZ[0] , newstream->rotationsXYZ[1] , newstream->rotationsXYZ[2]); }
*/
               unsigned int numberOfArguments = InputParser_GetNumberOfArguments(ipc);


               //fprintf(stderr,"TODO : TRAJECTORYPRIMITIVES_MOVE should ultimately just supply a 4x4 matrix to the levels below or offer an orientation order..");

               if (numberOfArguments==9)
               {
                 //We have received an euler angle rotation..
                 if (newstream->debug) { fprintf(stderr,"Rotation for object `%s` @ time %u is euler angles\n",name,time);}
                 coordLength=6;
               } else
               if (numberOfArguments==10)
               {
                 //We have received a quaternion rotation..
                 if (newstream->debug) { fprintf(stderr,"Rotation for object `%s` @ time %u is quaternion but converted to euler angles\n",name,time);}
                 coordLength=6;
                 quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];
                 convertQuaternionsToEulerAngles(newstream,euler,quaternions);
                 pos[3]=euler[0]; pos[4]=euler[1];  pos[5]=euler[2]; pos[6]=0.0;
               } else
               {
                fprintf(stderr,"Movement command for object `%s` has %u arguments and don't know its rotation order\n",name,InputParser_GetNumberOfArguments(ipc));
               }

                //fprintf(stderr,"Tracker POS OBJ( %f %f %f ,  %f %f %f )\n",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);
                addStateToObjectMini( newstream , name  , time , (float*) pos , coordLength );
          break; 

         case TRAJECTORYPRIMITIVES_LIGHT :
           fprintf(stderr,"Enabling light..\n");
           newstream->useLightingSystem=1;
           newstream->lightPosition[0]=InputParser_GetWordFloat(ipc,1);
           newstream->lightPosition[1]=InputParser_GetWordFloat(ipc,2);
           newstream->lightPosition[2]=InputParser_GetWordFloat(ipc,3);
         break;

          case TRAJECTORYPRIMITIVES_EVENT :
              if (InputParser_WordCompareNoCase(ipc,1,(char*)"INTERSECTS",10)==1)
                     { eventType = EVENT_INTERSECTION; }

              InputParser_GetWord(ipc,2,name,MAX_PATH);
              objIDA = getObjectID(newstream,name,&foundA);

              InputParser_GetWord(ipc,3,name,MAX_PATH);
              objIDB = getObjectID(newstream,name,&foundB);

              if ( (foundA) && (foundB) )
              {
               InputParser_GetWord(ipc,4,model,MAX_PATH);
               addEventToVirtualStream(newstream,objIDA,objIDB,eventType,model,InputParser_GetWordLength(ipc,4));
              }
          break;

          case TRAJECTORYPRIMITIVES_DONE_DECLARING_OBJECTS :
              doneAddingObjectsToVirtualStream(newstream);
          break;

          case TRAJECTORYPRIMITIVES_PQ :
               //PQ(ID,X,Y,Z,QX,QY,QZ,QW)
               item = InputParser_GetWordInt(ipc,1);
               //item+= + 1 + newstream->objDeclarationsOffset; /*Item 0 is camera so we +1 */

               frame = InputParser_GetWordInt(ipc,2);

               pos[0] = newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,3);
               pos[1] = newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,4);
               pos[2] = newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,5);
               pos[3] = InputParser_GetWordFloat(ipc,6);
               pos[4] = InputParser_GetWordFloat(ipc,7);
               pos[5] = InputParser_GetWordFloat(ipc,8);
               pos[6] = InputParser_GetWordFloat(ipc,9);
               if ( (pos[3]==0) && (pos[4]==0)  && (pos[5]==0)  && (pos[6]==0)  )
                  {
                    /*fprintf(stderr,"OBJ %u , frame %u declared with completely zero quaternion normalizing it to 0,0,0,1\n",item,newstream->timestamp);*/
                    pos[6]=1.0;
                  }


               coordLength=6;
               quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];
               convertQuaternionsToEulerAngles(newstream,euler,quaternions);
               pos[3]=euler[0]; pos[4]=euler[1];  pos[5]=euler[2]; pos[6]=0.0;


               addStateToObjectID( newstream , item , frame*newstream->rate  , (float*) pos , coordLength ,
                                   newstream->object[item].scaleX,
                                   newstream->object[item].scaleY,
                                   newstream->object[item].scaleZ,
                                   newstream->object[item].R,
                                   newstream->object[item].G,
                                   newstream->object[item].B,
                                   newstream->object[item].Transparency);

              // if ( (item==newstream->numberOfObjects) || (INCREMENT_TIMER_FOR_EACH_OBJ) ) { newstream->timestamp+=100; }


               #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker OBJ%u(now has %u / %u positions )\n",item,newstream->object[item].numberOfFrames,newstream->object[item].MAX_numberOfFrames);
               #endif
          break;










    default :

        if ( /*(!newstream->silent)||*/ (newstream->debug) )
            {
              fprintf(stderr,RED "Can't recognize `%s` \n" NORMAL , line);
            }
    break;
  };


            if (
                  ( InputParser_GetWordChar(ipc,0,0)=='O' ) &&
                  ( InputParser_GetWordChar(ipc,0,1)=='B' ) &&
                  ( InputParser_GetWordChar(ipc,0,2)=='J' ) &&
                  ( ( InputParser_GetWordChar(ipc,0,3)>='0' ) && ( InputParser_GetWordChar(ipc,0,3)<='9' )  )
               )
            {
               char name[MAX_PATH];
               InputParser_GetWord(ipc,0,name,MAX_PATH);
               char * itemNumStr = &name[3];

               unsigned int item = atoi(itemNumStr);  // (unsigned int) InputParser_GetWordChar(ipc,0,3)-'0';
               item+= + 1 + newstream->objDeclarationsOffset; /*Item 0 is camera so we +1 */

               float pos[7]={0};
               pos[0] = newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,1);
               pos[1] = newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,2);
               pos[2] = newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,3);
               pos[3] = InputParser_GetWordFloat(ipc,4);
               pos[4] = InputParser_GetWordFloat(ipc,5);
               pos[5] = InputParser_GetWordFloat(ipc,6);
               pos[6] = InputParser_GetWordFloat(ipc,7);
               if ( (pos[3]==0) && (pos[4]==0)  && (pos[5]==0)  && (pos[6]==0)  )
                  {
                    /*fprintf(stderr,"OBJ %u , frame %u declared with completely zero quaternion normalizing it to 0,0,0,1\n",item,newstream->timestamp);*/
                    pos[6]=1.0;
                  }

               int coordLength=7;

               float euler[3];
               float quaternions[4]; quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];

               normalizeQuaternions(&quaternions[0],&quaternions[1],&quaternions[2],&quaternions[3]);
               quaternions2Euler(euler,quaternions,1); //1
               pos[3] = newstream->rotationsOffset[0] + (newstream->scaleWorld[3] * euler[0]);
               pos[4] = newstream->rotationsOffset[1] + (newstream->scaleWorld[4] * euler[1]);
               pos[5] = newstream->rotationsOffset[2] + (newstream->scaleWorld[5] * euler[2]);
               pos[6] = 0;

               #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker OBJ%u( %f %f %f ,  %f %f %f )\n",item,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);
                fprintf(stderr,"Angle Offset %f %f %f \n",newstream->rotationsOffset[0],newstream->rotationsOffset[1],newstream->rotationsOffset[2]);
               #endif

               if (newstream->rotationsOverride)
                    { flipRotationAxis(&pos[3],&pos[4],&pos[5], newstream->rotationsXYZ[0] , newstream->rotationsXYZ[1] , newstream->rotationsXYZ[2]); }





               coordLength=6;
               quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];
               convertQuaternionsToEulerAngles(newstream,euler,quaternions);
               pos[3]=euler[0]; pos[4]=euler[1];  pos[5]=euler[2]; pos[6]=0.0;


               addStateToObjectID( newstream , item , newstream->timestamp , (float*) pos , coordLength ,
                                   newstream->object[item].scaleX,
                                   newstream->object[item].scaleY,
                                   newstream->object[item].scaleZ,
                                   newstream->object[item].R,
                                   newstream->object[item].G,
                                   newstream->object[item].B,
                                   newstream->object[item].Transparency);

               if ( (item==newstream->numberOfObjects) || (INCREMENT_TIMER_FOR_EACH_OBJ) ) { newstream->timestamp+=100; }


               #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker OBJ%u(now has %u / %u positions )\n",item,newstream->object[item].numberOfFrames,newstream->object[item].MAX_numberOfFrames);
               #endif
            } else
            /*! REACHED A POSITION DECLERATION ( ARROWX(103.0440706,217.1741961,-22.9230451,0.780506107461,0.625148155413,-0,0.00285155239622) )
              argument 0 = ARROW , argument 1-3 = X/Y/Z ,  argument 4-7 =  ux/uy/uz  , argument 8 Scale */
            if (
                  ( InputParser_GetWordChar(ipc,0,0)=='A' ) &&
                  ( InputParser_GetWordChar(ipc,0,1)=='R' ) &&
                  ( InputParser_GetWordChar(ipc,0,2)=='R' ) &&
                  ( InputParser_GetWordChar(ipc,0,3)=='O' ) &&
                  ( InputParser_GetWordChar(ipc,0,4)=='W' ) &&
                  ( ( InputParser_GetWordChar(ipc,0,5)>='0' ) && ( InputParser_GetWordChar(ipc,0,5)<='9' )  )
               )
            {

               char name[MAX_PATH];
               InputParser_GetWord(ipc,0,name,MAX_PATH);
               char * itemNumStr = &name[5];

               unsigned int item = atoi(itemNumStr);  // (unsigned int) InputParser_GetWordChar(ipc,0,5)-'0';
               item+= + 1 + newstream->objDeclarationsOffset; /*Item 0 is camera so we +1 */

               InputParser_GetWord(ipc,1,name,MAX_PATH);

               float pos[7]={0};
               pos[0] = newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,1);
               pos[1] = newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,2);
               pos[2] = newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,3);
               //float deltaX = InputParser_GetWordFloat(ipc,4);
               //float deltaY = InputParser_GetWordFloat(ipc,5);
               //float deltaZ = InputParser_GetWordFloat(ipc,6);
               float scale = InputParser_GetWordFloat(ipc,8);
               pos[3] = 0.0; // newstream->scaleWorld[3] * InputParser_GetWordFloat(ipc,6);
               pos[4] = 0.0; // newstream->scaleWorld[4] * InputParser_GetWordFloat(ipc,7);
               pos[5] = 0.0; // newstream->scaleWorld[5] * InputParser_GetWordFloat(ipc,8);
               pos[6] = 1.0; // InputParser_GetWordFloat(ipc,9);
               int coordLength=7;

               pos[3] = InputParser_GetWordFloat(ipc,4);
               pos[4] = InputParser_GetWordFloat(ipc,5);
               pos[5] = InputParser_GetWordFloat(ipc,6);
               pos[6] = InputParser_GetWordFloat(ipc,7);

               float euler[3];
               float quaternions[4]; quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];

               normalizeQuaternions(&quaternions[0],&quaternions[1],&quaternions[2],&quaternions[3]);
               quaternions2Euler(euler,quaternions,1); //1
               pos[3] = newstream->rotationsOffset[0] + (newstream->scaleWorld[3] * euler[0]);
               pos[4] = newstream->rotationsOffset[1] + (newstream->scaleWorld[4] * euler[1]);
               pos[5] = newstream->rotationsOffset[2] + (newstream->scaleWorld[5] * euler[2]);
               pos[6] = 0;


               #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker ARROW%u( %f %f %f ,  %f %f %f )\n",item,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);
                fprintf(stderr,"Angle Offset %f %f %f \n",newstream->rotationsOffset[0],newstream->rotationsOffset[1],newstream->rotationsOffset[2]);
               #endif

               if (newstream->rotationsOverride)
                    { flipRotationAxis(&pos[3],&pos[4],&pos[5], newstream->rotationsXYZ[0] , newstream->rotationsXYZ[1] , newstream->rotationsXYZ[2]); }


               addStateToObjectID( newstream , item , newstream->timestamp , (float*) pos , coordLength ,
                                   newstream->object[item].scaleX * scale , //Arrows scale only X dimension
                                   newstream->object[item].scaleY, //<- i could also add scales here
                                   newstream->object[item].scaleZ, //<- i could also add scales here
                                   newstream->object[item].R ,
                                   newstream->object[item].G ,
                                   newstream->object[item].B ,
                                   newstream->object[item].Transparency
                                   );

               if ( (item==newstream->numberOfObjects) || (INCREMENT_TIMER_FOR_EACH_OBJ) ) { newstream->timestamp+=100; }

               #if PRINT_LOAD_INFO
                fprintf(stderr,"Tracker ARROW%u(now has %u / %u positions )\n",item,newstream->object[item].numberOfFrames,newstream->object[item].MAX_numberOfFrames);
               #endif
            }  else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"HAND_POINTS0",12)==1)
            {
               char curItem[128];
               unsigned int item=0;
               float pos[7]={0};
               int coordLength=6;

               fprintf(stderr,"Trying to parse Hand Points\n");
               unsigned int found = 0;

               int i=0;
               for (i=0; i<23; i++)
               {
                sprintf(curItem,"hand%u_sph%u",0,i);
                item = getObjectID(newstream, curItem, &found );
                if (found)
                 {
                  pos[0]=newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,1+i*3);
                  pos[1]=newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,2+i*3);
                  pos[2]=newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,3+i*3);
                  addStateToObjectID( newstream , item , newstream->timestamp , (float*) pos , coordLength ,
                                      newstream->object[item].scaleX,
                                      newstream->object[item].scaleY,
                                      newstream->object[item].scaleZ,
                                      newstream->object[item].R,
                                      newstream->object[item].G,
                                      newstream->object[item].B,
                                      newstream->object[item].Transparency);
                 }
               }
            }

 return 1;
}




int appendVirtualStreamFromFile(struct VirtualStream * newstream , struct ModelList * modelStorage,const char * filename)
{
  #warning "Code of readVirtualStream should be rewritten..! :P"
  //#warning "This should probably be split down to some primitives and also support things like including a file from another file"
  //#warning "dynamic reload of models/objects explicit support for Quaternions / Rotation Matrices and getting rid of some intermediate"
  //#warning "parser declerations like arrowsX or objX"

  #if USE_FILE_INPUT
  //Our stack variables ..
  unsigned int fileSize=0;
  unsigned int readOpResult = 0;
  char line [LINE_MAX_LENGTH]={0};

  //Try and open filename
  FILE * fp = fopen(filename,"r");
  if (fp == 0 ) { fprintf(stderr,"Cannot open trajectory stream %s \n",filename); return 0; }

  //Allocate a token parser
  struct InputParserC * ipc=0;
  ipc = InputParser_Create(LINE_MAX_LENGTH,5);
  if (ipc==0)  { fprintf(stderr,"Cannot allocate memory for new stream\n"); fclose(fp); return 0; }

 //Everything is set , Lets read the file!
  while (!feof(fp))
  {
   //We get a new line out of the file
   readOpResult = (fgets(line,LINE_MAX_LENGTH,fp)!=0);
   if ( readOpResult != 0 )
    {
      //We tokenize it
      unsigned int words_count = InputParser_SeperateWords(ipc,line,0);
      if ( words_count > 0 )
         {
             unsigned int label=scanFor_TrajectoryPrimitives(line,strlen(line));
             processCommand(newstream,modelStorage,ipc,label,line,words_count);
         } // End of line containing tokens
    } //End of getting a line while reading the file
  }

  fclose(fp);
  InputParser_Destroy(ipc);



  return 1;
  #else
    fprintf(stderr,RED "This build of Trajectory parser does not have File Input compiled in!\n" NORMAL);
    fprintf(stderr,YELLOW "Please rebuild after setting USE_FILE_INPUT to 1 on file TrajectoryParser.cpp or .c\n" NORMAL);
    fprintf(stderr,YELLOW "also note that by enabling file input you will also need to link with InputParser_C.cpp or .c \n" NORMAL);
    fprintf(stderr,YELLOW "It can be found here https://github.com/AmmarkoV/InputParser/ \n" NORMAL);
  #endif

 return 0;
}



int readVirtualStream(struct VirtualStream * newstream,struct ModelList * modelStorage)
{
  //Try and open filename to get the size ( this is needed for auto refresh functionality to work correctly..!
   FILE * fp = fopen(newstream->filename,"r");
   if (fp == 0 ) { fprintf(stderr,"Cannot open trajectory stream %s \n",newstream->filename); return 0; } else
    {
      fseek (fp , 0 , SEEK_END);
      unsigned long lSize = ftell (fp);
      rewind (fp);
      fprintf(stderr,"Opening a %lu byte file %s \n",lSize,newstream->filename);
      newstream->fileSize = lSize;
    }

  //Do initial state here , make sure we will start reading using a clean state
  growVirtualStreamObjectsTypes(newstream,OBJECT_TYPES_TO_ADD_STEP);
  strcpy( newstream->objectTypes[0].name , "camera" );
  strcpy( newstream->objectTypes[0].model , "camera" );
  ++newstream->numberOfObjectTypes;

  growVirtualStreamObjects(newstream,OBJECTS_TO_ADD_STEP);
  strcpy( newstream->object[0].name, "camera");
  strcpy( newstream->object[0].typeStr, "camera");
  strcpy( newstream->object[0].value, "camera");
  newstream->object[0].type = 0; //Camera
  newstream->object[0].R =0;
  newstream->object[0].G =0;
  newstream->object[0].B =0;
  newstream->scaleWorld[0]=1.0; newstream->scaleWorld[1]=1.0; newstream->scaleWorld[2]=1.0;
  newstream->scaleWorld[3]=1.0; newstream->scaleWorld[4]=1.0; newstream->scaleWorld[5]=1.0;
  newstream->object[0].Transparency=0;
  ++newstream->numberOfObjects;
  // CAMERA OBJECT ADDED

  newstream->objDeclarationsOffset=0;
  newstream->rotationsOverride=0;
  newstream->rotationsXYZ[0]=0; newstream->rotationsXYZ[1]=1; newstream->rotationsXYZ[2]=2;
  newstream->rotationsOffset[0]=0.0; newstream->rotationsOffset[1]=0.0; newstream->rotationsOffset[2]=0.0;

  newstream->debug=0;

  //We refresh our associated model Storage
   newstream->associatedModelList = modelStorage;


  fclose(fp);
  return appendVirtualStreamFromFile(newstream,modelStorage,newstream->filename);
}



int destroyVirtualStreamInternal(struct VirtualStream * stream,int also_destrstream_struct)
{
   if (stream==0) { return 1; }
   if (stream->object==0) { return 1; }
   unsigned int i =0 ;

  //CLEAR OBJECTS , AND THEIR FRAMES
   for ( i=0; i<stream->MAX_numberOfObjects; i++)
    {
       if ( stream->object[i].frame!= 0 )
         {
            free(stream->object[i].frame);
            stream->object[i].frame=0;
         }
    }
   stream->MAX_numberOfObjects=0;
   stream->numberOfObjects=0;
   free(stream->object);
   stream->object=0;

   //CLEAR TYPES OF OBJECTS
    if ( stream->objectTypes!= 0 )
         {
            free(stream->objectTypes);
            stream->objectTypes=0;
         }
    stream->MAX_numberOfObjectTypes=0;
    stream->numberOfObjectTypes=0;


  hashMap_Destroy(stream->objectHash);
  hashMap_Destroy(stream->objectTypesHash);
  hashMap_Destroy(stream->connectorHash);
  hashMap_Destroy(stream->eventHash);



   if (also_destrstream_struct) { free(stream); }
   return 1;
}


int destroyVirtualStream(struct VirtualStream * stream)
{
    return destroyVirtualStreamInternal(stream,1);
}



int refreshVirtualStream(struct VirtualStream * newstream,struct ModelList * modelStorage)
{
   #if PRINT_DEBUGGING_INFO
   fprintf(stderr,"refreshingVirtualStream\n");
   #endif

   destroyVirtualStreamInternal(newstream,0);
   //Please note that the newstream structure does not get a memset operation anywhere around here
   //thats in order to keep the initial time / frame configuration
   //Object numbers , Object type numbers,  Frame numbers are cleaned by the destroyVirtualStreamInternal call

   //We refresh our associated model Storage
   newstream->associatedModelList = modelStorage;

   return readVirtualStream(newstream,modelStorage);
}


int setVirtualStreamDefaults(struct VirtualStream * scene)
{
  scene->rate=1.0;


  scene->controls.selectedOBJ=0;
  scene->controls.framesRendered=0;


  scene->controls.tickUSleepTime=100;
  scene->controls.pauseTicking=0;
  scene->controls.farPlane = 255; //<--be aware that this has an effect on the depth maps generated
  scene->controls.nearPlane= 1; //<--this also
  scene->controls.fieldOfView = 65;
  scene->controls.scaleDepthTo =1000.0;

  scene->controls.moveSpeed=0.5;

  scene->controls.lastRenderingTime=0;
  scene->controls.lastFramerate=0;

  //float depthUnit = 1.0;

  scene->controls.userKeyFOVEnabled=0;
  return 1;
}



struct VirtualStream * createVirtualStream(const char * filename , struct ModelList * modelStorage)
{
  //Allocate a virtual stream structure
  struct VirtualStream * newstream = (struct VirtualStream *) malloc(sizeof(struct VirtualStream));
  if (newstream==0)  {  fprintf(stderr,"Cannot allocate memory for new stream\n"); return 0; }


  //Clear the whole damn thing..
  memset(newstream,0,sizeof(struct VirtualStream));

  //Set values that are needed
  //newstream->rate=1.0;
  setVirtualStreamDefaults(newstream);

  int useSorting=0;
  newstream->objectHash = hashMap_Create(200,200,0,useSorting);
  newstream->objectTypesHash = hashMap_Create(200,200,0,0);
  #warning "Setting hashmap for object types to use sorting does not work well, Is there a bug >"
  newstream->connectorHash = hashMap_Create(200,200,0,useSorting);
  newstream->eventHash = hashMap_Create(200,200,0,useSorting);

  //We refresh our associated model Storage
   newstream->associatedModelList = modelStorage;


  if (filename!=0)
  {

  fprintf(stderr,"strncpy from %p to %p \n",filename,newstream->filename);
   //strncpy(newstream->filename,filename,MAX_PATH);
     myStrCpy(newstream->filename,filename,MAX_PATH);
  fprintf(stderr,"strncpy returned\n");

   if (!readVirtualStream(newstream,modelStorage))
    {
      fprintf(stderr,"Could not read Virtual Stream from file %s \n",filename);
      destroyVirtualStream(newstream);
      return 0;
    }
  } else
  {
    fprintf(stderr,"Created an empty virtual stream\n");
  }

  return newstream;
}

