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
#include "../../../../tools/AmMatrix/matrixCalculations.h"
//Using normalizeQuaternionsTJP #include "../../../../tools/AmMatrix/matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>




#if USE_FILE_INPUT
  #include "InputParser_C.h"
#endif


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


//int (*saveSnapshot) (int,struct calibration *);


int writeVirtualStream(struct VirtualStream * newstream,char * filename)
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
              (int) newstream->object[i].R/255,
              (int) newstream->object[i].G/255,
              (int) newstream->object[i].B/255,
              (int) newstream->object[i].Transparency/255,
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




int appendVirtualStreamFromFile(struct VirtualStream * newstream , char * filename)
{
  #warning "Code of readVirtualStream is *quickly* turning to shit after a chain of unplanned insertions on the parser"
  #warning "This should probably be split down to some primitives and also support things like including a file from another file"
  #warning "dynamic reload of models/objects explicit support for Quaternions / Rotation Matrices and getting rid of some intermediate"
  #warning "parser declerations like arrowsX or objX"

  #if USE_FILE_INPUT
  //Our stack variables ..
  unsigned int fileSize=0;
  unsigned int readOpResult = 0;
  char line [LINE_MAX_LENGTH]={0};

  //Try and open filename
  FILE * fp = fopen(filename,"r");
  if (fp == 0 ) { fprintf(stderr,"Cannot open trajectory stream %s \n",filename); return 0; }

  //Find out the size of the file , This is no longer needed..!
  /*
  fseek (fp , 0 , SEEK_END);
  unsigned long lSize = ftell (fp);
  rewind (fp);
  fprintf(stderr,"Opening a %lu byte file %s \n",lSize,filename);
  fileSize = lSize;
  */

  //Allocate a token parser
  struct InputParserC * ipc=0;
  ipc = InputParser_Create(LINE_MAX_LENGTH,5);
  if (ipc==0)  { fprintf(stderr,"Cannot allocate memory for new stream\n"); return 0; }

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

               double euler[3];
               double quaternions[4]; quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];

               normalizeQuaternionsTJP(&quaternions[0],&quaternions[1],&quaternions[2],&quaternions[3]);
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
            }
              else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"FRAME_RESET",11)==1)
            {
               newstream->timestamp=0;  //Reset Frame
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"FRAME",5)==1)
            {
               newstream->timestamp+=100; //Increment Frame
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES",32)==1)
            {
               unsigned int satteliteObj = 1 + newstream->objDeclarationsOffset + InputParser_GetWordInt(ipc,1);    /*Item 0 is camera so we +1 */
               unsigned int planetObj    = 1 + newstream->objDeclarationsOffset + InputParser_GetWordInt(ipc,2);    /*Item 0 is camera so we +1 */
               unsigned int frame     = InputParser_GetWordInt(ipc,3);
               unsigned int duration  = InputParser_GetWordInt(ipc,4);
               if (! affixSatteliteToPlanetFromFrameForLength(newstream,satteliteObj,planetObj,frame,duration) )
               {
                fprintf(stderr,RED "Could not affix Object %u to Object %u for %u frames ( starting @ %u )\n" NORMAL , satteliteObj,planetObj,duration,frame);
               }
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"INCLUDE",7)==1)
            {
               char includeFile[MAX_PATH]={0};
               InputParser_GetWord(ipc,1,includeFile,MAX_PATH);
              if (appendVirtualStreamFromFile(newstream,includeFile))
              {
                fprintf(stderr,GREEN "Successfully included file %s..!" NORMAL,includeFile);
              } else
              {
                fprintf(stderr,RED "Could not include include file..!" NORMAL);
              }

            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"DEBUG",5)==1)
            {
              fprintf(stderr,"DEBUG Mode on\n");
              newstream->debug=1;
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"MOVE_VIEW",9)==1)
            {
              newstream->userCanMoveCameraOnHisOwn=InputParser_GetWordInt(ipc,1);
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"TIMESTAMP",9)==1)
            {
              newstream->timestamp=InputParser_GetWordInt(ipc,1);
            } else
            /*! REACHED A SMOOTH DECLERATION ( SMOOTH() )  */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"SMOOTH",6)==1)
            {
              smoothTrajectories(newstream);
            } else

            /*! REACHED A SMOOTH DECLERATION ( SMOOTH() )  */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"OBJ_OFFSET",10)==1)
            {
              newstream->objDeclarationsOffset = InputParser_GetWordInt(ipc,1);
            } else
            /*! REACHED AN AUTO REFRESH DECLERATION ( AUTOREFRESH(1500) )
              argument 0 = AUTOREFRESH , argument 1 = value in milliseconds (0 = off ) */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"AUTOREFRESH",11)==1)
            {
                newstream->autoRefresh = InputParser_GetWordInt(ipc,1);
            } else
            /*! REACHED AN INTERPOLATE TIME SWITCH DECLERATION ( INTERPOLATE_TIME(1) )
              argument 0 = INTERPOLATE_TIME , argument 1 = (0 = off ) ( 1 = on )*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"INTERPOLATE_TIME",16)==1)
            {
                //The configuration INTERPOLATE_TIME is the "opposite" of this flag ignore time
                newstream->ignoreTime = InputParser_GetWordInt(ipc,1);
                // so we flip it here.. , the default is not ignoring time..
                if (newstream->ignoreTime == 0 ) { newstream->ignoreTime=1; } else
                                                 { newstream->ignoreTime=0; }
            } else
              /*! REACHED AN BACKGROUND DECLERATION ( BACKGROUND(0,0,0) )
              argument 0 = BACKGROUND , argument 1 = R , argument 2 = G , argument 3 = B , */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"BACKGROUND",10)==1)
            {
                //The configuration INTERPOLATE_TIME is the "opposite" of this flag ignore time
                newstream->backgroundR = (float) InputParser_GetWordInt(ipc,1) / 255;
                newstream->backgroundG = (float) InputParser_GetWordInt(ipc,2) / 255;
                newstream->backgroundB = (float) InputParser_GetWordInt(ipc,3) / 255;
                // so we flip it here.. , the default is not ignoring time..
            } else
            /*! REACHED AN OBJECT TYPE DECLERATION ( OBJECTTYPE(spatoula_type,"spatoula.obj") )
              argument 0 = OBJECTTYPE , argument 1 = name ,  argument 2 = value */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"OBJECTTYPE",10)==1)
            {
               char name[MAX_PATH]={0};
               char model[MAX_PATH]={0};
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,model,MAX_PATH);

               addObjectTypeToVirtualStream( newstream , name, model );

            } else
            /*! REACHED A CONNECTOR DECLERATION ( CONNECTOR(something,somethingElse,0,255,0,0,1.0,type) )
              argument 0 = CONNECTOR , argument 1 = nameOfFirstObject ,  argument 2 = nameOfSecondObject ,  argument 3-5 = RGB color  , argument 6 Transparency , argument 7 = Scale , argument 8 = Type */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"CONNECTOR",9)==1)
            {
               char firstObject[MAX_PATH]={0} , secondObject[MAX_PATH]={0} , type[MAX_PATH]={0};
               InputParser_GetWord(ipc,1,firstObject,MAX_PATH);
               InputParser_GetWord(ipc,2,secondObject,MAX_PATH);

               unsigned char R = (unsigned char) InputParser_GetWordInt(ipc,3);
               unsigned char G = (unsigned char)  InputParser_GetWordInt(ipc,4);
               unsigned char B = (unsigned char)  InputParser_GetWordInt(ipc,5);
               unsigned char Alpha = (unsigned char)  InputParser_GetWordInt(ipc,6);
               float scale = (float) InputParser_GetWordFloat(ipc,7);

               addConnectorToVirtualStream(
                                            newstream ,
                                            firstObject , secondObject,
                                            R, G , B , Alpha ,
                                            scale,
                                            type
                                          );

            } else
            /*! REACHED AN OBJECT DECLERATION ( OBJECT(something,spatoula_type,0,255,0,0,0,1.0,spatoula_something) )
              argument 0 = OBJECT , argument 1 = name ,  argument 2 = type ,  argument 3-5 = RGB color  , argument 6 Transparency , argument 7 = No Color ,
              argument 8 = ScaleX , argument 9 = ScaleY , argument 10 = ScaleZ , argument 11 = String Freely formed Data */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"OBJECT",6)==1)
            {
               char name[MAX_PATH]={0} , typeStr[MAX_PATH]={0};
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,typeStr,MAX_PATH);

               unsigned char R = (unsigned char) InputParser_GetWordInt(ipc,3);
               unsigned char G = (unsigned char)  InputParser_GetWordInt(ipc,4);
               unsigned char B = (unsigned char)  InputParser_GetWordInt(ipc,5);
               unsigned char Alpha = (unsigned char)  InputParser_GetWordInt(ipc,6) ;
               unsigned char nocolor = (unsigned char) InputParser_GetWordInt(ipc,7);
               float scaleX = (float) InputParser_GetWordFloat(ipc,8);
               float scaleY = (float) InputParser_GetWordFloat(ipc,9);
               float scaleZ = (float) InputParser_GetWordFloat(ipc,10);

               //Value , not used : InputParser_GetWord(ipc,8,newstream->object[pos].value,15);
               addObjectToVirtualStream(newstream ,name,typeStr,R,G,B,Alpha,nocolor,0,0,scaleX,scaleY,scaleZ,0);

            } else
            /*! REACHED AN COMPOSITEOBJECT DECLERATION ( COMPOSITEOBJECT(something,spatoula_type,0,255,0,0,0,1.0,1.0,1.0,27,spatoula_something) )
              argument 0 = COMPOSITEOBJECT , argument 1 = name ,  argument 2 = type ,  argument 3-5 = RGB color  , argument 6 Transparency , argument 7 = No Color ,
              argument 8 = ScaleX , argument 9 = ScaleY , argument 10 = ScaleZ , argument 11 = Number of arguments , argument 12 = String Freely formed Data */
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"COMPOSITEOBJECT",15)==1)
            {
               char name[MAX_PATH]={0} , typeStr[MAX_PATH]={0};
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               InputParser_GetWord(ipc,2,typeStr,MAX_PATH);

               unsigned char R = (unsigned char) InputParser_GetWordInt(ipc,3);
               unsigned char G = (unsigned char)  InputParser_GetWordInt(ipc,4);
               unsigned char B = (unsigned char)  InputParser_GetWordInt(ipc,5);
               unsigned char Alpha = (unsigned char)  InputParser_GetWordInt(ipc,6) ;
               unsigned char nocolor = (unsigned char) InputParser_GetWordInt(ipc,7);
               float scaleX = (float) InputParser_GetWordFloat(ipc,8);
               float scaleY = (float) InputParser_GetWordFloat(ipc,9);
               float scaleZ = (float) InputParser_GetWordFloat(ipc,10);
               unsigned int numberOfParticles = (unsigned char)  InputParser_GetWordInt(ipc,11);

               //Value , not used : InputParser_GetWord(ipc,8,newstream->object[pos].value,15);
               addObjectToVirtualStream(newstream ,name,typeStr,R,G,B,Alpha,nocolor,0,0,scaleX,scaleY,scaleZ,numberOfParticles);

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

               double euler[3];
               double quaternions[4]; quaternions[0]=pos[3]; quaternions[1]=pos[4]; quaternions[2]=pos[5]; quaternions[3]=pos[6];

               normalizeQuaternionsTJP(&quaternions[0],&quaternions[1],&quaternions[2],&quaternions[3]);
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

            /*! REACHED A POSITION DECLERATION ( POS(hand,0,   0.0,0.0,0.0 , 0.0,0.0,0.0,0.0 ) )
              argument 0 = POS , argument 1 = name ,  argument 2 = time in MS , argument 3-5 = X,Y,Z , argument 6-9 = Rotations*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"POS",3)==1)
            {
               char name[MAX_PATH];
               InputParser_GetWord(ipc,1,name,MAX_PATH);
               unsigned int time = InputParser_GetWordInt(ipc,2);

               float pos[7]={0};
               pos[0] = newstream->scaleWorld[0] * InputParser_GetWordFloat(ipc,3);
               pos[1] = newstream->scaleWorld[1] * InputParser_GetWordFloat(ipc,4);
               pos[2] = newstream->scaleWorld[2] * InputParser_GetWordFloat(ipc,5);
               pos[3] = newstream->scaleWorld[3] * InputParser_GetWordFloat(ipc,6);
               pos[4] = newstream->scaleWorld[4] * InputParser_GetWordFloat(ipc,7);
               pos[5] = newstream->scaleWorld[5] * InputParser_GetWordFloat(ipc,8);
               pos[6] = InputParser_GetWordFloat(ipc,9);
               int coordLength=7;

               if (newstream->rotationsOverride)
                     { flipRotationAxis(&pos[3],&pos[4],&pos[5], newstream->rotationsXYZ[0] , newstream->rotationsXYZ[1] , newstream->rotationsXYZ[2]); }

               unsigned int found = 0;
               unsigned int item = getObjectID(newstream, name, &found );
               if (found)
               {
                //fprintf(stderr,"Tracker POS OBJ( %f %f %f ,  %f %f %f )\n",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);
                addStateToObjectID( newstream , item  , time , (float*) pos , coordLength,
                                    newstream->object[item].scaleX,
                                    newstream->object[item].scaleY,
                                    newstream->object[item].scaleZ,
                                    newstream->object[item].R,
                                    newstream->object[item].G,
                                    newstream->object[item].B,
                                    newstream->object[item].Transparency );
               } else
               {
                 fprintf(stderr,RED "Could not add state/position to non-existing object `%s` \n" NORMAL,name);
               }
            }

             else
            /*! REACHED A PROJECTION MATRIX DECLERATION ( PROJECTION_MATRIX( ... 16 values ... ) )
              argument 0 = PROJECTION_MATRIX , argument 1-16 matrix values*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"EVENT",5)==1)
            {

              unsigned int eventType=0;
               if (InputParser_WordCompareNoCase(ipc,1,(char*)"INTERSECTS",10)==1)
                     {
                       eventType = EVENT_INTERSECTION;
                     }

              unsigned int foundA = 0 , foundB = 0;
              char name[MAX_PATH];
              InputParser_GetWord(ipc,2,name,MAX_PATH);
              unsigned int objIDA = getObjectID(newstream,name,&foundA);

              InputParser_GetWord(ipc,3,name,MAX_PATH);
              unsigned int objIDB = getObjectID(newstream,name,&foundB);

              if ( (foundA) && (foundB) )
              {
               char buf[256];
               InputParser_GetWord(ipc,4,buf,256);
               if (addEventToVirtualStream(newstream,objIDA,objIDB,eventType,buf,InputParser_GetWordLength(ipc,4)) )
               {
                 fprintf(stderr,"addedEvent\n");
               } else
               {
                 fprintf(stderr,"Could NOT add event\n");
               }
              }
            }
             else
            /*! REACHED A PROJECTION MATRIX DECLERATION ( PROJECTION_MATRIX( ... 16 values ... ) )
              argument 0 = PROJECTION_MATRIX , argument 1-16 matrix values*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"PROJECTION_MATRIX",17)==1)
            {
               newstream->projectionMatrixDeclared=1;
               int i=1;
               for (i=1; i<=16; i++) { newstream->projectionMatrix[i-1] = (double)  InputParser_GetWordFloat(ipc,i); }
               fprintf(stderr,"Projection Matrix given to TrajectoryParser\n");
            }
             else
            /*! REACHED AN EMULATE PROJECTION MATRIX DECLERATION ( PROJECTION_MATRIX( ... 9 values ... ) )
              argument 0 = PROJECTION_MATRIX , argument 1-9 matrix values*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"EMULATE_PROJECTION_MATRIX",25)==1)
            {
               newstream->emulateProjectionMatrixDeclared=1;
               int i=1;
               for (i=1; i<=9; i++) { newstream->emulateProjectionMatrix[i-1] = (double)  InputParser_GetWordFloat(ipc,i); }
               fprintf(stderr,"Emulating Projection Matrix given to TrajectoryParser\n");
            }
             else
            /*! REACHED A MODELVIEW MATRIX DECLERATION ( MODELVIEW_MATRIX( ... 16 values ... ) )
              argument 0 = MODELVIEW_MATRIX , argument 1-16 matrix values*/
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"MODELVIEW_MATRIX",16)==1)
            {
               newstream->modelViewMatrixDeclared=1;
               int i=1;
               for (i=1; i<=16; i++) { newstream->modelViewMatrix[i-1] = (double) InputParser_GetWordFloat(ipc,i); }
               fprintf(stderr,"ModelView Matrix given to TrajectoryParser\n");
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"SCALE_WORLD",11)==1)
            {
               newstream->scaleWorld[0] = InputParser_GetWordFloat(ipc,1);
               newstream->scaleWorld[1] = InputParser_GetWordFloat(ipc,2);
               newstream->scaleWorld[2] = InputParser_GetWordFloat(ipc,3);
               fprintf(stderr,"Scaling everything * %f %f %f \n",newstream->scaleWorld[0],newstream->scaleWorld[1],newstream->scaleWorld[2]);
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"OFFSET_ROTATIONS",16)==1)
            {
               newstream->rotationsOffset[0] = InputParser_GetWordFloat(ipc,1);
               newstream->rotationsOffset[1] = InputParser_GetWordFloat(ipc,2);
               newstream->rotationsOffset[2] = InputParser_GetWordFloat(ipc,3);
            } else
            if (InputParser_WordCompareNoCase(ipc,0,(char*)"MAP_ROTATIONS",13)==1)
            {
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

            }

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



int readVirtualStream(struct VirtualStream * newstream)
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

  return appendVirtualStreamFromFile(newstream,newstream->filename);
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

   if (also_destrstream_struct) { free(stream); }
   return 1;
}


int destroyVirtualStream(struct VirtualStream * stream)
{
    return destroyVirtualStreamInternal(stream,1);
}



int refreshVirtualStream(struct VirtualStream * newstream)
{
   #if PRINT_DEBUGGING_INFO
   fprintf(stderr,"refreshingVirtualStream\n");
   #endif

   destroyVirtualStreamInternal(newstream,0);
   //Please note that the newstream structure does not get a memset operation anywhere around here
   //thats in order to keep the initial time / frame configuration
   //Object numbers , Object type numbers,  Frame numbers are cleaned by the destroyVirtualStreamInternal call

   return readVirtualStream(newstream);
}




void myStrCpy(char * destination,char * source,unsigned int maxDestinationSize)
{
  unsigned int i=0;
  while ( (i<maxDestinationSize) && (source[i]!=0) ) { destination[i]=source[i]; ++i; }
}

struct VirtualStream * createVirtualStream(char * filename)
{
  //Allocate a virtual stream structure
  struct VirtualStream * newstream = (struct VirtualStream *) malloc(sizeof(struct VirtualStream));
  if (newstream==0)  {  fprintf(stderr,"Cannot allocate memory for new stream\n"); return 0; }


  //Clear the whole damn thing..
  memset(newstream,0,sizeof(struct VirtualStream));

  if (filename!=0)
  {

  fprintf(stderr,"strncpy from %p to %p \n",filename,newstream->filename);
   //strncpy(newstream->filename,filename,MAX_PATH);
     myStrCpy(newstream->filename,filename,MAX_PATH);
  fprintf(stderr,"strncpy returned\n");

   if (!readVirtualStream(newstream))
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

/*!
    ------------------------------------------------------------------------------------------
                       /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\
                                 READING FILES , CREATING CONTEXT
    ------------------------------------------------------------------------------------------

    ------------------------------------------------------------------------------------------
                                     GETTING AN OBJECT POSITION
                       \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/
    ------------------------------------------------------------------------------------------

*/



