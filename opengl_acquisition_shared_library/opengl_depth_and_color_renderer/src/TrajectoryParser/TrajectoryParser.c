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
#include "../../../../tools/AmMatrix/matrixCalculations.h"
//Using normalizeQuaternionsTJP #include "../../../../tools/AmMatrix/matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define PI 3.141592653589793238462643383279502884197

#define LINE_MAX_LENGTH 1024
#define OBJECT_TYPES_TO_ADD_STEP 10
#define OBJECTS_TO_ADD_STEP 10
#define EVENTS_TO_ADD_STEP 10
#define FRAMES_TO_ADD_STEP 123

#define PRINT_DEBUGGING_INFO 0
#define PRINT_WARNING_INFO 0
#define PRINT_LOAD_INFO 0

#define CASE_SENSITIVE_OBJECT_NAMES 0

#define USE_QUATERNIONS_FOR_ORBITING 0

//If you want Trajectory parser to be able to READ
//and parse files you should set  USE_FILE_INPUT  to 1
#define USE_FILE_INPUT 1
//-------------------------------------------------------

#if USE_FILE_INPUT
  #include "InputParser_C.h"
#endif

//This is retarded , i have to remake parsing to fix this
#define INCREMENT_TIMER_FOR_EACH_OBJ 0

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


//int (*saveSnapshot) (int,struct calibration *);


/*!
    ------------------------------------------------------------------------------------------
                       /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\
                                 GROWING MEMORY ALLOCATIONS
    ------------------------------------------------------------------------------------------

    ------------------------------------------------------------------------------------------
                                 SEARCHING OF OBJECT ID's , TYPES etc
                       \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/
    ------------------------------------------------------------------------------------------

*/

static int dummy_strcasecmp_internal(char * input1, char * input2)
{
  #if CASE_SENSITIVE_OBJECT_NAMES
    return strcmp(input1,input2);
  #endif

    if ( (input1==0) || (input2==0) )
     {
         fprintf(stderr,"Error , calling strcasecmp_internal with null parameters \n");
         return 1;
     }
    unsigned int len1 = strlen(input1);
    unsigned int len2 = strlen(input2);
    if (len1!=len2)
     {
         //mismatched lengths of strings , they can't be equal..!
         return 1;
     }

   char A; //<- character buffer for input1
   char B; //<- character buffer for input2
   unsigned int i=0;
   while (i<len1) //len1 and len2 are equal
    {
       A = toupper(input1[i]);
       B = toupper(input2[i]);
       if (A!=B) { return 1; }
       ++i;
    }
  //if we reached this point , there where no reasons
  //why input1 and input2 could not be equal..
  return 0;
}


void listAllObjectTypeID(struct VirtualStream * stream)
{
  fprintf(stderr,"Listing all declared ObjectTypeIDs ( %u )---------------\n",stream->numberOfObjectTypes);
  unsigned int i=0;
  for (i=0; i<stream->numberOfObjectTypes; i++ ) { fprintf(stderr,"%u - %s => %s \n",i,stream->objectTypes[i].name,stream->objectTypes[i].model); }
}

ObjectIDHandler getObjectID(struct VirtualStream * stream,char * name, unsigned int * found)
{
  if (stream==0) { fprintf(stderr,"Can't get object id (%s) for un allocated stream\n",name); }
  if (stream->object==0) { fprintf(stderr,"Can't get object id (%s) for un allocated object array\n",name); }

  *found=0;
  unsigned int i=0;
  for (i=0; i<stream->numberOfObjects; i++ )
   {
       if (dummy_strcasecmp_internal(name,stream->object[i].name)==0)
         {
              *found=1;
              return i;
         }
   }

   return 0;
}

ObjectTypeID getObjectTypeID(struct VirtualStream * stream,char * typeName,unsigned int * found)
{
  if (stream==0) { fprintf(stderr,"Can't get object id (%s) for un allocated stream\n",typeName); }
  if (stream->objectTypes==0) { fprintf(stderr,"Can't get object id (%s) for un allocated object type array\n",typeName); }

  *found=0;
  unsigned int i=0;
  for (i=0; i<stream->numberOfObjectTypes; i++ )
   {
       if (dummy_strcasecmp_internal(typeName,stream->objectTypes[i].name)==0)
         {
              *found=1;
              return i;
         }
         //else { fprintf(stderr,"ObjType `%s` != `%s` requested \n",stream->objectTypes[i].name,typeName); }
   }

   return 0;
}

char * getObjectTypeModel(struct VirtualStream * stream,ObjectTypeID typeID)
{
  if (stream==0) { fprintf(stderr,"Can't get object id (%u) for un allocated stream\n",typeID); return 0; }
  if (stream->objectTypes==0) { fprintf(stderr,"Can't get object id (%u) for un allocated object type array\n",typeID); return 0;  }
  if (typeID>=stream->numberOfObjectTypes) { fprintf(stderr,"Can't get object id (%u) we only got %u Object Types \n",typeID,stream->numberOfObjectTypes); return 0; }

  return stream->objectTypes[typeID].model;
}

char * getModelOfObjectID(struct VirtualStream * stream,ObjectIDHandler ObjID)
{
  if (stream==0) { fprintf(stderr,"Can't get object (%u) for un allocated stream\n",ObjID); return 0; }
  if (stream->object==0) { fprintf(stderr,"Can't get model of object id (%u) for un allocated object array\n",ObjID); return 0;  }
  if (stream->objectTypes==0) { fprintf(stderr,"Can't get model of object id (%u) for un allocated object type array\n",ObjID); return 0;  }
  if (ObjID >= stream->numberOfObjects ) { fprintf(stderr,"Can't get object id (%u) we only got %u Objects \n", ObjID , stream->numberOfObjects); return 0; }

  ObjectTypeID typeID = stream->object[ObjID].type;
  if (typeID>=stream->numberOfObjectTypes) { fprintf(stderr,"Can't get object id (%u) we only got %u Object Types \n",typeID,stream->numberOfObjectTypes); return 0; }

  return stream->objectTypes[typeID].model;
}

int getObjectColorsTrans(struct VirtualStream * stream,ObjectIDHandler ObjID,float * R,float * G,float * B,float * Transparency, unsigned char * noColor)
{
  *R = stream->object[ObjID].R;
  *G = stream->object[ObjID].G;
  *B = stream->object[ObjID].B;
  *Transparency = stream->object[ObjID].Transparency;
  *noColor = stream->object[ObjID].nocolor;
  return 1;
}

/*!
    ------------------------------------------------------------------------------------------
                       /\   /\   /\   /\   /\   /\   /\   /\   /\   /\   /\
                                 SEARCHING OF OBJECT ID's , TYPES etc
    ------------------------------------------------------------------------------------------

    ------------------------------------------------------------------------------------------
                                   READING FILES , CREATING CONTEXT
                       \/   \/   \/   \/   \/   \/   \/   \/   \/   \/   \/
    ------------------------------------------------------------------------------------------

*/

unsigned long getFileSize(char * filename)
{
  FILE * fp = fopen(filename,"r");
  if (fp == 0 ) { fprintf(stderr,"getFileSize cannot open %s \n",filename); return 0; }

  //Find out the size of the file..!
  fseek (fp , 0 , SEEK_END);
  unsigned long lSize = ftell (fp);
  fclose(fp);

  return lSize;
}



int addStateToObjectID(
                               struct VirtualStream * stream ,
                               unsigned int ObjID  ,
                               unsigned int timeMilliseconds ,
                               float * coord ,
                               unsigned int coordLength ,
                               float scaleX , float scaleY ,float scaleZ ,
                               float R , float G , float B , float Alpha
                       )
{
  if (stream->object[ObjID].MAX_numberOfFrames<=stream->object[ObjID].numberOfFrames+1) { growVirtualStreamFrames(&stream->object[ObjID],FRAMES_TO_ADD_STEP); }
  //Now we should definately have enough space for our new frame
  if (stream->object[ObjID].MAX_numberOfFrames<=stream->object[ObjID].numberOfFrames+1) { fprintf(stderr,"Cannot add new POS instruction to Object %u \n",ObjID); return 0; }

  //We have the space so lets fill our new frame spot ..!
  unsigned int pos = stream->object[ObjID].numberOfFrames;
  ++stream->object[ObjID].numberOfFrames;

  // 1 is object name
  stream->object[ObjID].frame[pos].time = timeMilliseconds;
  stream->object[ObjID].frame[pos].isQuaternion = 0;

  stream->object[ObjID].frame[pos].scaleX = scaleX;
  stream->object[ObjID].frame[pos].scaleY = scaleY;
  stream->object[ObjID].frame[pos].scaleZ = scaleZ;
  stream->object[ObjID].frame[pos].R = R;
  stream->object[ObjID].frame[pos].G = G;
  stream->object[ObjID].frame[pos].B = B;
  stream->object[ObjID].frame[pos].Alpha = Alpha;

  if (coordLength > 0 ) {  stream->object[ObjID].frame[pos].x = coord[0]; }
  if (coordLength > 1 ) {  stream->object[ObjID].frame[pos].y = coord[1]; }
  if (coordLength > 2 ) {  stream->object[ObjID].frame[pos].z = coord[2]; }

  if (coordLength > 3 ) {stream->object[ObjID].frame[pos].rot1 = coord[3]; }
  if (coordLength > 4 ) {stream->object[ObjID].frame[pos].rot2 = coord[4]; }
  if (coordLength > 5 ) {stream->object[ObjID].frame[pos].rot3 = coord[5]; }
  if (coordLength > 6 ) {stream->object[ObjID].frame[pos].rot4 = coord[6]; stream->object[ObjID].frame[pos].isQuaternion = 1; }

  if (stream->object[ObjID].MAX_timeOfFrames <= stream->object[ObjID].frame[pos].time)
    {
     stream->object[ObjID].MAX_timeOfFrames = stream->object[ObjID].frame[pos].time;
    } else
    {
     fprintf(stderr,"Error in configuration file , object positions not in correct time order (this %u , last max %u).. \n",
             stream->object[ObjID].frame[pos].time,
             stream->object[ObjID].MAX_timeOfFrames);
    }

  #if PRINT_DEBUGGING_INFO
   fprintf(stderr,"String %s resolves to : \n",line);
   fprintf(stderr,"X %02f Y %02f Z %02f ROT %02f %02f %02f %02f\n",stream->object[ObjID].frame[pos].x,stream->object[ObjID].frame[pos].y,stream->object[ObjID].frame[pos].z ,
   stream->object[ObjID].frame[pos].rot1 , stream->object[ObjID].frame[pos].rot2 , stream->object[ObjID].frame[pos].rot3 , stream->object[ObjID].frame[pos].rot4 );
  #endif

  return 1;
}


int addStateToObject(
                              struct VirtualStream * stream ,
                              char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength ,
                              float scaleX , float scaleY ,float scaleZ ,
                              float R , float G , float B , float Alpha
                       )
{

 unsigned int ObjFound = 0;
 unsigned int ObjID = getObjectID(stream,name,&ObjFound);
 if (ObjFound)
  {
     return addStateToObjectID(stream,ObjID,timeMilliseconds,coord,coordLength,scaleX,scaleY,scaleZ,R,G,B,Alpha);
  }
  fprintf(stderr,"Could not Find object %s \n",name);
  return 0;
}



int addConnectorToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * firstObject , char * secondObject ,
                                 unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                                 float scale,
                                 char * typeStr
                               )
{
   if (stream->MAX_numberOfConnectors<=stream->numberOfConnectors+1) { growVirtualStreamConnectors(stream,OBJECTS_TO_ADD_STEP); }
   //Now we should definately have enough space for our new frame
   if (stream->MAX_numberOfConnectors<=stream->numberOfConnectors+1) { fprintf(stderr,"Cannot add new OBJECT instruction\n"); return 0; }

   unsigned int found=0;
   unsigned int pos = stream->numberOfConnectors;

   strcpy(stream->connector[pos].firstObject,firstObject);
   stream->connector[pos].objID_A = getObjectID(stream,firstObject,&found);
   if (!found) { fprintf(stderr,"Couldn't find object id for object %s \n",firstObject); }

   strcpy(stream->connector[pos].secondObject,secondObject);
   stream->connector[pos].objID_B = getObjectID(stream,secondObject,&found);
   if (!found) { fprintf(stderr,"Couldn't find object id for object %s \n",secondObject); }

   strcpy(stream->connector[pos].typeStr,typeStr);
   stream->connector[pos].R = (float) R/255;
   stream->connector[pos].G = (float) G/255;
   stream->connector[pos].B = (float) B/255;
   stream->connector[pos].Transparency = (float) Alpha/100;
   stream->connector[pos].scale = scale;

   ++stream->numberOfConnectors;
   return 1;
}

int addObjectToVirtualStream(
                              struct VirtualStream * stream ,
                              char * name , char * type ,
                              unsigned char R, unsigned char G , unsigned char B , unsigned char Alpha ,
                              unsigned char noColor ,
                              float * coords ,
                              unsigned int coordLength ,
                              float scaleX,
                              float scaleY,
                              float scaleZ,
                              unsigned int particleNumber
                            )
{
   if (stream->MAX_numberOfObjects<=stream->numberOfObjects+1) { growVirtualStreamObjects(stream,OBJECTS_TO_ADD_STEP); }
   //Now we should definately have enough space for our new frame
   if (stream->MAX_numberOfObjects<=stream->numberOfObjects+1) { fprintf(stderr,"Cannot add new OBJECT instruction\n"); return 0; }

   //We have the space so lets fill our new object spot ..!
   unsigned int pos = stream->numberOfObjects;

   //Clearing  everything is done in growVirtualStreamObjects so no need to do it here
   //memset((void*) &stream->object[pos],0,sizeof(struct VirtualObject));

   strcpy(stream->object[pos].name,name);
   strcpy(stream->object[pos].typeStr,type);
   stream->object[pos].R = (float) R/255;
   stream->object[pos].G = (float) G/255;
   stream->object[pos].B = (float) B/255;
   stream->object[pos].Transparency = (float) Alpha/100;
   stream->object[pos].nocolor = noColor;
   stream->object[pos].scaleX = scaleX;
   stream->object[pos].scaleY = scaleY;
   stream->object[pos].scaleZ = scaleZ;

   if ( (scaleX==0.0) || (scaleY==0.0) || (scaleZ==0.0) )
   {
       fprintf(stderr,RED "Please note that scaling parameters (%f,%f,%f) will effectively make object %s invisible \n" NORMAL,scaleX,scaleY,scaleZ,name);
   }

   stream->object[pos].particleNumber = particleNumber;

   stream->object[pos].frame=0;

  #warning "Check this part of the code "
  //<-------- check from here ------------->
  stream->object[pos].lastFrame=0; // <- todo here <- check these
  stream->object[pos].MAX_numberOfFrames=0;
  stream->object[pos].numberOfFrames=0;
  //<-------- check up until here ------------->

   unsigned int found=0;
   stream->object[pos].type = getObjectTypeID(stream,stream->object[pos].typeStr,&found);
   if (!found) {
                 fprintf(stderr,"Please note that type %s couldn't be found for object %s \n",stream->object[pos].typeStr,stream->object[pos].name);
                 //listAllObjectTypeID(stream);
               }

   fprintf(stderr,"addedObject(%s,%s) with ID %u ,typeID %u \n",name,type,pos,stream->object[pos].type);
   ++stream->numberOfObjects;


   if (coords!=0)
   {
    if (! addStateToObject(stream,name,0,coords,coordLength,
                           stream->object[pos].scaleX,
                           stream->object[pos].scaleY,
                           stream->object[pos].scaleZ,
                           stream->object[pos].R,
                           stream->object[pos].G,
                           stream->object[pos].B,
                           stream->object[pos].Transparency) )
    {
       fprintf(stderr,"Cannot add initial position to new object\n");
    }
   }



   return 1; // <- we always return
   return found;
}


int removeObjectFromVirtualStream(struct VirtualStream * stream , unsigned int ObjID )
{
 fprintf(stderr,"removeObjectFromVirtualStream is a stub , it is not implemented , ObjID %u stayed in stream (%p) \n",ObjID,stream);
 return 0;
}


int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 char * type , char * model
                                )
{
    if (stream->MAX_numberOfObjectTypes<=stream->numberOfObjectTypes+1) { growVirtualStreamObjectsTypes(stream,OBJECT_TYPES_TO_ADD_STEP); }
    //Now we should definately have enough space for our new frame
    if (stream->MAX_numberOfObjectTypes<=stream->numberOfObjectTypes+1) { fprintf(stderr,"Cannot add new OBJECTTYPE instruction\n"); }

    //We have the space so lets fill our new object spot ..!
    unsigned int pos = stream->numberOfObjectTypes;
    strcpy(stream->objectTypes[pos].name,type);
    strcpy(stream->objectTypes[pos].model,model);

    fprintf(stderr,"addedObjectType(%s,%s) with ID %u \n",type,model,pos);

    ++stream->numberOfObjectTypes;

    return 1; // <- we a
}


int addEventToVirtualStream(
                             struct VirtualStream * stream ,
                             unsigned int objIDA ,
                             unsigned int objIDB ,
                             unsigned int eventType ,
                             char * data ,
                             unsigned int dataSize
                           )
{
    if (stream->MAX_numberOfEvents<=stream->numberOfEvents+1) { growVirtualStreamEvents(stream,EVENTS_TO_ADD_STEP); }
    //Now we should definately have enough space for our new frame
    if (stream->MAX_numberOfEvents<=stream->numberOfEvents+1) { fprintf(stderr,"Cannot add new Event instruction\n"); }

    //We have the space so lets fill our new object spot ..!
    unsigned int pos = stream->numberOfEvents;
    stream->event[pos].objID_A=objIDA;
    stream->event[pos].objID_B=objIDB;
    stream->event[pos].eventType = eventType;

    stream->event[pos].dataSize = dataSize;
    stream->event[pos].data = (char *) malloc((dataSize+1) * sizeof(char));
    memcpy(stream->event[pos].data,data,dataSize);
    stream->event[pos].data[dataSize]=0;
/*
    int i=0;
    i=system(data);
    if (i==0) { fprintf(stderr,"lel\n"); }*/

    ++stream->numberOfEvents;

    return 1; // <- we a
}


int smoothTrajectoriesOfObject(struct VirtualStream * stream,unsigned int ObjID)
{
  float avg=0.0;
  unsigned int pos=0;
  for (pos=1; pos<stream->object[ObjID].numberOfFrames; pos++)
  {
    //------------------------------------------------------------------------------------
    avg = stream->object[ObjID].frame[pos-1].x + stream->object[ObjID].frame[pos].x;
    stream->object[ObjID].frame[pos-1].x = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].y + stream->object[ObjID].frame[pos].y;
    stream->object[ObjID].frame[pos-1].y = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].z + stream->object[ObjID].frame[pos].z;
    stream->object[ObjID].frame[pos-1].z = avg / 2;
    //------------------------------------------------------------------------------------


    //------------------------------------------------------------------------------------
    avg = stream->object[ObjID].frame[pos-1].rot1 + stream->object[ObjID].frame[pos].rot1;
    stream->object[ObjID].frame[pos-1].rot1 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot2 + stream->object[ObjID].frame[pos].rot2;
    stream->object[ObjID].frame[pos-1].rot2 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot3 + stream->object[ObjID].frame[pos].rot3;
    stream->object[ObjID].frame[pos-1].rot3 = avg / 2;

    avg = stream->object[ObjID].frame[pos-1].rot4 + stream->object[ObjID].frame[pos].rot4;
    stream->object[ObjID].frame[pos-1].rot4 = avg / 2;
    //------------------------------------------------------------------------------------
  }
 return 1;
}

int smoothTrajectories(struct VirtualStream * stream)
{
  fprintf(stderr,"Smoothing %u objects \n",stream->numberOfObjects);
  unsigned int objID=0;
  for (objID=0; objID<stream->numberOfObjects; objID++)
  {
     smoothTrajectoriesOfObject(stream,objID);
  }
 return 1;
}



float calculateDistanceTra(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z)
{
   float vect_x = from_x - to_x;
   float vect_y = from_y - to_y;
   float vect_z = from_z - to_z;

   return  (sqrt(pow(vect_x, 2) + pow(vect_y, 2) + pow(vect_z, 2)));

}

void euler2QuaternionsInternal(double * quaternions,double * euler,int quaternionConvention)
{
  //This conversion follows the rule euler X Y Z  to quaternions W X Y Z
  //Our input is degrees so we convert it to radians for the sin/cos functions
  double eX = (double) (euler[0] * PI) / 180;
  double eY = (double) (euler[1] * PI) / 180;
  double eZ = (double) (euler[2] * PI) / 180;

  //fprintf(stderr,"eX %f eY %f eZ %f\n",eX,eY,eZ);

  //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  //eX Roll  φ - rotation about the X-axis
  //eY Pitch θ - rotation about the Y-axis
  //eZ Yaw   ψ - rotation about the Z-axis

  double cosX2 = cos((double) eX/2); //cos(φ/2);
  double sinX2 = sin((double) eX/2); //sin(φ/2);
  double cosY2 = cos((double) eY/2); //cos(θ/2);
  double sinY2 = sin((double) eY/2); //sin(θ/2);
  double cosZ2 = cos((double) eZ/2); //cos(ψ/2);
  double sinZ2 = sin((double) eZ/2); //sin(ψ/2);

  switch (quaternionConvention )
  {
   case 1 :
   /*qX*/ quaternions[0] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[1] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[2] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   /*qW*/ quaternions[3] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   break;

   case 0 :
   /*qW*/ quaternions[0] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   /*qX*/ quaternions[1] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[2] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[3] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   break;

   default :
    fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
   break;
  };

}

#if USE_QUATERNIONS_FOR_ORBITING
int affixSatteliteToPlanetFromFrameForLength(struct VirtualStream * stream,unsigned int satteliteObj,unsigned int planetObj , unsigned int frameNumber , unsigned int duration)
{
    //There is literally no good reason to go from rotation -> quaternion -> 3x3 -> quaternion -> rotation this could be optimized
    //==================================================================================
    double satPosAbsolute[4]={0};
    satPosAbsolute[0] = (double) stream->object[satteliteObj].frame[frameNumber].x;
    satPosAbsolute[1] = (double) stream->object[satteliteObj].frame[frameNumber].y;
    satPosAbsolute[2] = (double) stream->object[satteliteObj].frame[frameNumber].z;
    satPosAbsolute[3] = 1.0;

    //==================================================================================
    double planetPosAbsolute[4]={0};
    planetPosAbsolute[0] = (double) stream->object[planetObj].frame[frameNumber].x;
    planetPosAbsolute[1] = (double) stream->object[planetObj].frame[frameNumber].y;
    planetPosAbsolute[2] = (double) stream->object[planetObj].frame[frameNumber].z;
    planetPosAbsolute[3] = 1.0;


    double planetQuatAbsolute[4]={0};
    double planetRotAbsolute[4]={0};
    double planetRotAbsoluteF[4]={0};
    planetRotAbsolute[0] = (double) stream->object[planetObj].frame[frameNumber].rot1;
    planetRotAbsolute[1] = (double) stream->object[planetObj].frame[frameNumber].rot2;
    planetRotAbsolute[2] = (double) stream->object[planetObj].frame[frameNumber].rot3;
    euler2QuaternionsInternal(planetQuatAbsolute , planetRotAbsolute,1);


    double satPosRelative[4]={0};
    pointFromAbsoluteToRelationWithObject_PosXYZQuaternionXYZW(0,satPosRelative,planetPosAbsolute,planetQuatAbsolute,satPosAbsolute);

    unsigned int pos=0;
    for (pos=frameNumber+1; pos<frameNumber+duration; pos++)
    {
       planetPosAbsolute[0] = (double) stream->object[planetObj].frame[pos].x;
       planetPosAbsolute[1] = (double) stream->object[planetObj].frame[pos].y;
       planetPosAbsolute[2] = (double) stream->object[planetObj].frame[pos].z;
       planetPosAbsolute[3] = 1.0;

       planetRotAbsoluteF[0] = stream->object[planetObj].frame[pos].rot1;
       planetRotAbsoluteF[1] = stream->object[planetObj].frame[pos].rot2;
       planetRotAbsoluteF[2] = stream->object[planetObj].frame[pos].rot3;

       //Undo all the evil that has been done to our coordinate system
       if (stream->rotationsOverride)
         {
            unflipRotationAxis(
                              &planetRotAbsoluteF[0],
                              &planetRotAbsoluteF[1],
                              &planetRotAbsoluteF[2],
                              stream->rotationsXYZ[0] ,
                              stream->rotationsXYZ[1] ,
                              stream->rotationsXYZ[2]
                              );
         }

       planetRotAbsolute[0] = (double) planetRotAbsoluteF[0];
       planetRotAbsolute[1] = (double) planetRotAbsoluteF[1];
       planetRotAbsolute[2] = (double) planetRotAbsoluteF[2];

       planetRotAbsolute[0] -= stream->rotationsOffset[0];
       planetRotAbsolute[1] -= stream->rotationsOffset[1];
       planetRotAbsolute[2] -= stream->rotationsOffset[2];
       planetRotAbsolute[0] =  planetRotAbsolute[0] / stream->scaleWorld[3];
       planetRotAbsolute[1] =  planetRotAbsolute[1] / stream->scaleWorld[4];
       planetRotAbsolute[2] =  planetRotAbsolute[2] / stream->scaleWorld[5];

       euler2QuaternionsInternal(planetQuatAbsolute , planetRotAbsolute,1);

       planetQuatAbsolute[0]=(-1) * planetQuatAbsolute[0];
       //planetQuatAbsolute[3]=(-1) * planetQuatAbsolute[3];

       if ( pointFromRelationWithObjectToAbsolute_PosXYZQuaternionXYZW(satPosAbsolute,planetPosAbsolute,planetQuatAbsolute,satPosRelative) )
       {
           stream->object[satteliteObj].frame[pos].x = (float) satPosAbsolute[0];
           stream->object[satteliteObj].frame[pos].y = (float) satPosAbsolute[1];
           stream->object[satteliteObj].frame[pos].z = (float) satPosAbsolute[2];
       }
    }
 return 1;
}
#else
int affixSatteliteToPlanetFromFrameForLength(struct VirtualStream * stream,unsigned int satteliteObj,unsigned int planetObj , unsigned int frameNumber , unsigned int duration)
{
    //There is literally no good reason to go from rotation -> quaternion -> 3x3 -> quaternion -> rotation this could be optimized
    //==================================================================================
    double satPosAbsolute[4]={0};
    satPosAbsolute[0] = (double) stream->object[satteliteObj].frame[frameNumber].x;
    satPosAbsolute[1] = (double) stream->object[satteliteObj].frame[frameNumber].y;
    satPosAbsolute[2] = (double) stream->object[satteliteObj].frame[frameNumber].z;
    satPosAbsolute[3] = 1.0;
    //==================================================================================
    double planetPosAbsolute[4]={0};
    planetPosAbsolute[0] = (double) stream->object[planetObj].frame[frameNumber].x;
    planetPosAbsolute[1] = (double) stream->object[planetObj].frame[frameNumber].y;
    planetPosAbsolute[2] = (double) stream->object[planetObj].frame[frameNumber].z;
    planetPosAbsolute[3] = 1.0;

    double planetRotAbsolute[4]={0};
    planetRotAbsolute[0] = (double) stream->object[planetObj].frame[frameNumber].rot1;
    planetRotAbsolute[1] = (double) stream->object[planetObj].frame[frameNumber].rot2;
    planetRotAbsolute[2] = (double) stream->object[planetObj].frame[frameNumber].rot3;
    //==================================================================================


    double satPosRelative[4]={0};
    pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(1,satPosRelative,planetPosAbsolute,planetRotAbsolute,satPosAbsolute);

    unsigned int pos=0;
    for (pos=frameNumber+1; pos<frameNumber+duration; pos++)
    {
       planetPosAbsolute[0] = (double) stream->object[planetObj].frame[pos].x;
       planetPosAbsolute[1] = (double) stream->object[planetObj].frame[pos].y;
       planetPosAbsolute[2] = (double) stream->object[planetObj].frame[pos].z;
       planetPosAbsolute[3] = 1.0;

       planetRotAbsolute[0] = (double) stream->object[planetObj].frame[pos].rot1;
       planetRotAbsolute[1] = (double) stream->object[planetObj].frame[pos].rot2;
       planetRotAbsolute[2] = (double) stream->object[planetObj].frame[pos].rot3;

       if ( pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(satPosAbsolute,planetPosAbsolute,planetRotAbsolute,satPosRelative) )
       {
           stream->object[satteliteObj].frame[pos].x = (float) satPosAbsolute[0];
           stream->object[satteliteObj].frame[pos].y = (float) satPosAbsolute[1];
           stream->object[satteliteObj].frame[pos].z = (float) satPosAbsolute[2];
       }
    }
 return 1;
}

#endif // USE_QUATERNIONS_FOR_ORBITING


int objectsCollide(struct VirtualStream * newstream,unsigned int atTime,unsigned int objIDA,unsigned int objIDB)
{
  float posA[7]={0}; float scaleA_X,scaleA_Y,scaleA_Z;
  float posB[7]={0}; float scaleB_X,scaleB_Y,scaleB_Z;

  calculateVirtualStreamPos(newstream,objIDA,atTime,posA,&scaleA_X,&scaleA_Y,&scaleA_Z);
  calculateVirtualStreamPos(newstream,objIDB,atTime,posB,&scaleB_X,&scaleB_Y,&scaleB_Z);

  float distance =  calculateDistanceTra(posA[0],posA[1],posA[2],posB[0],posB[1],posB[2]);
  fprintf(stderr,"Distance %u from %u = %f\n",objIDA,objIDB,distance);
  if ( distance > 0.3 ) { return 0;}

  return 1;
}



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

int normalizeQuaternionsTJP(double *qX,double *qY,double *qZ,double *qW)
{
#if USE_FAST_NORMALIZATION
      // Works best when quat is already almost-normalized
      double f = (double) (3.0 - (((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)))) / 2.0;
      *qX *= f;
      *qY *= f;
      *qZ *= f;
      *qW *= f;
#else
      double sqrtDown = (double) sqrt(((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)));
      double f = (double) 1 / sqrtDown;
       *qX *= f;
       *qY *= f;
       *qZ *= f;
       *qW *= f;
#endif // USE_FAST_NORMALIZATION
  return 1;
}

void quaternions2Euler(double * euler,double * quaternions,int quaternionConvention)
{
    double qX,qY,qZ,qW;

    euler[0]=0.0; euler[1]=0.0; euler[2]=0.0;

    switch (quaternionConvention)
     {
       case 0  :
       qW = quaternions[0];
       qX = quaternions[1];
       qY = quaternions[2];
       qZ = quaternions[3];
       break;

       case 1 :
       qX = quaternions[0];
       qY = quaternions[1];
       qZ = quaternions[2];
       qW = quaternions[3];
       break;

       default :
       fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
       break;
     }

  //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  //e1 Roll  - rX: rotation about the X-axis
  //e2 Pitch - rY: rotation about the Y-axis
  //e3 Yaw   - rZ: rotation about the Z-axis

  //Shorthand to go according to http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles
  double q0=qW , q1 = qX , q2 = qY , q3 = qZ;
  double q0q1 = (double) q0*q1 , q2q3 = (double) q2*q3;
  double q0q2 = (double) q0*q2 , q3q1 = (double) q3*q1;
  double q0q3 = (double) q0*q3 , q1q2 = (double) q1*q2;


  double eXDenominator = ( 1.0 - 2.0 * (q1*q1 + q2*q2) );
  if (eXDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }
  double eYDenominator = ( 1.0 - 2.0 * ( q2*q2 + q3*q3) );
  if (eYDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }


  /* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
     We need to replace the arctan by atan2 to generate all the orientations. */
  /*eX*/ euler[0] = atan2( (2.0 *  (q0q1 + q2q3)) , eXDenominator ) ;
  /*eY*/ euler[1] = asin( 2.0 * (q0q2 - q3q1));
  /*eZ*/ euler[2] = atan2( (2.0 * (q0q3 + q1q2)) ,  eYDenominator );

  //Our output is in radians so we convert it to degrees for the user

  //Go from radians back to degrees
  euler[0] = (euler[0] * 180) / PI;
  euler[1] = (euler[1] * 180) / PI;
  euler[2] = (euler[2] * 180) / PI;

}


int flipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ)
{
  #if PRINT_LOAD_INFO
   fprintf(stderr,"Had rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
   fprintf(stderr,"Moving 0 to %u , 1 to %u , 2 to %u \n",where2SendX,where2SendY,where2SendZ);
  #endif

  float tmpX = *rotX;
  float tmpY = *rotY;
  float tmpZ = *rotZ;
  //-----------------------------------------
  if (where2SendX==0) { *rotX=tmpX; } else
  if (where2SendX==1) { *rotY=tmpX; } else
  if (where2SendX==2) { *rotZ=tmpX; }

  if (where2SendY==0) { *rotX=tmpY; } else
  if (where2SendY==1) { *rotY=tmpY; } else
  if (where2SendY==2) { *rotZ=tmpY; }

  if (where2SendZ==0) { *rotX=tmpZ; } else
  if (where2SendZ==1) { *rotY=tmpZ; } else
  if (where2SendZ==2) { *rotZ=tmpZ; }
  //-----------------------------------------

  #if PRINT_LOAD_INFO
   fprintf(stderr,"Now have rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
  #endif

  return 1;
}



int unflipRotationAxis(float * rotX, float * rotY , float * rotZ , int where2SendX , int where2SendY , int where2SendZ)
{
  #warning "Is   unflipRotationAxis correct ? "
  #if PRINT_LOAD_INFO
   fprintf(stderr,"Had rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
   fprintf(stderr,"Moving 0 to %u , 1 to %u , 2 to %u \n",where2SendX,where2SendY,where2SendZ);
  #endif

  float tmpX = 0;
  float tmpY = 0;
  float tmpZ = 0;
  //-----------------------------------------
  if (where2SendX==0) { tmpX=*rotX; } else
  if (where2SendX==1) { tmpX=*rotY; } else
  if (where2SendX==2) { tmpX=*rotZ; }

  if (where2SendY==0) { tmpY=*rotX; } else
  if (where2SendY==1) { tmpY=*rotY; } else
  if (where2SendY==2) { tmpY=*rotZ; }

  if (where2SendZ==0) { tmpZ=*rotX; } else
  if (where2SendZ==1) { tmpZ=*rotY; } else
  if (where2SendZ==2) { tmpZ=*rotZ; }
  //-----------------------------------------

  *rotX=tmpX;
  *rotY=tmpY;
  *rotZ=tmpZ;

  #if PRINT_LOAD_INFO
   fprintf(stderr,"Now have rotX %f rotY %f rotZ %f \n",*rotX,*rotY,*rotZ);
  #endif

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





int fillPosWithNull(float * pos,float * scaleX ,float * scaleY,float * scaleZ)
{
    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning null frame for obj %u \n",ObjID);
    #endif

    pos[0]=0.0;
    pos[1]=0.0;
    pos[2]=0.0;
    pos[3]=0.0;
    pos[4]=0.0;
    pos[5]=0.0;
    pos[6]=0.0;
    *scaleX = 1.0;
    *scaleY = 1.0;
    *scaleZ = 1.0;

    return 1;
}


int fillPosWithLastFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ )
{
   if (stream==0) { fprintf(stderr,"Cannot fill position on empty stream \n"); return 0; }
   if (pos==0) { fprintf(stderr,"Cannot fill position on empty position \n"); return 0; }
   if (ObjID>=stream->numberOfObjects) { fprintf(stderr,"Trying to add position for a non existing object\n"); return 0; }

   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
       fprintf(stderr,"Cannot Access frames for object %u \n",ObjID);
      #endif
      return 0;
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning frame %u \n",FrameIDToReturn);
    #endif

    unsigned int FrameIDToReturn = stream->object[ObjID].numberOfFrames;
    if (FrameIDToReturn>0) { --FrameIDToReturn; } //We have FrameIDToReturn frames so we grab the last one ( FrameIDToReturn -1 )
    pos[0]=stream->object[ObjID].frame[FrameIDToReturn].x;
    pos[1]=stream->object[ObjID].frame[FrameIDToReturn].y;
    pos[2]=stream->object[ObjID].frame[FrameIDToReturn].z;
    pos[3]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    pos[4]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    pos[5]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    pos[6]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
    *scaleX=stream->object[ObjID].frame[FrameIDToReturn].scaleX;
    *scaleY=stream->object[ObjID].frame[FrameIDToReturn].scaleY;
    *scaleZ=stream->object[ObjID].frame[FrameIDToReturn].scaleZ;
    return 1;
}

/*
int fillPosWithLastFrameD(struct VirtualStream * stream,ObjectIDHandler ObjID,double * pos,double * scale )
{
   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
      fprintf(stderr,"Cannot Access frames for object %u \n",ObjID);
      #endif
      return 0;
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning frame %u \n",FrameIDToReturn);
    #endif

    unsigned int FrameIDToReturn = stream->object[ObjID].numberOfFrames;
    if (FrameIDToReturn>0) { --FrameIDToReturn; } //We have FrameIDToReturn frames so we grab the last one ( FrameIDToReturn -1 )
    pos[0]=(double) stream->object[ObjID].frame[FrameIDToReturn].x;
    pos[1]=(double) stream->object[ObjID].frame[FrameIDToReturn].y;
    pos[2]=(double) stream->object[ObjID].frame[FrameIDToReturn].z;
    pos[3]=(double) stream->object[ObjID].frame[FrameIDToReturn].rot1;
    pos[4]=(double) stream->object[ObjID].frame[FrameIDToReturn].rot2;
    pos[5]=(double) stream->object[ObjID].frame[FrameIDToReturn].rot3;
    pos[6]=(double) stream->object[ObjID].frame[FrameIDToReturn].rot4;
    *scale=stream->object[ObjID].frame[FrameIDToReturn].scale;
    return 1;
}*/


int fillPosWithFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int FrameIDToReturn,float * pos,float * scaleX,float * scaleY,float * scaleZ)
{
   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
      fprintf(stderr,"Cannot Access frames for object %u \n",ObjID);
      #endif
      return 0;
    }

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Returning frame %u \n",FrameIDToReturn);
    #endif

    if (FrameIDToReturn >= stream->object[ObjID].MAX_numberOfFrames )
     {
         fprintf(stderr,"fillPosWithFrame asked to return frame out of bounds ( %u / %u / %u Max ) \n",FrameIDToReturn,stream->object[ObjID].numberOfFrames,stream->object[ObjID].MAX_numberOfFrames);
         return 0;
     }

    pos[0]=stream->object[ObjID].frame[FrameIDToReturn].x;
    pos[1]=stream->object[ObjID].frame[FrameIDToReturn].y;
    pos[2]=stream->object[ObjID].frame[FrameIDToReturn].z;
    pos[3]=stream->object[ObjID].frame[FrameIDToReturn].rot1;
    pos[4]=stream->object[ObjID].frame[FrameIDToReturn].rot2;
    pos[5]=stream->object[ObjID].frame[FrameIDToReturn].rot3;
    pos[6]=stream->object[ObjID].frame[FrameIDToReturn].rot4;
    *scaleX=stream->object[ObjID].frame[FrameIDToReturn].scaleX;
    *scaleY=stream->object[ObjID].frame[FrameIDToReturn].scaleY;
    *scaleZ=stream->object[ObjID].frame[FrameIDToReturn].scaleZ;
    return 1;
}


int fillPosWithInterpolatedFrame(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ,
                                 unsigned int PrevFrame,unsigned int NextFrame , unsigned int time )
{
   if (stream->object[ObjID].frame==0)
    {
      #if PRINT_WARNING_INFO
      fprintf(stderr,"Cannot Access interpolated frames for object %u \n",ObjID);
      #endif
      return 0;
    }

   if (PrevFrame==NextFrame)
    {
       return fillPosWithFrame(stream,ObjID,PrevFrame,pos,scaleX,scaleY,scaleZ);
    }


    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"Interpolating frames @  %u , between %u and %u \n",time,PrevFrame,NextFrame);
    #endif
    float interPos[7]={0};
    float interScale;

    unsigned int MAX_stepTime= stream->object[ObjID].frame[NextFrame].time - stream->object[ObjID].frame[PrevFrame].time;
    if (MAX_stepTime == 0 ) { MAX_stepTime=1; }
    unsigned int our_stepTime= time - stream->object[ObjID].frame[PrevFrame].time;


    interPos[0]=(float) ( stream->object[ObjID].frame[NextFrame].x-stream->object[ObjID].frame[PrevFrame].x ) * our_stepTime / MAX_stepTime;
    interPos[0]+=stream->object[ObjID].frame[PrevFrame].x;

    interPos[1]=(float) ( stream->object[ObjID].frame[NextFrame].y-stream->object[ObjID].frame[PrevFrame].y ) * our_stepTime / MAX_stepTime;
    interPos[1]+=stream->object[ObjID].frame[PrevFrame].y;

    interPos[2]=(float) ( stream->object[ObjID].frame[NextFrame].z-stream->object[ObjID].frame[PrevFrame].z ) * our_stepTime / MAX_stepTime;
    interPos[2]+=stream->object[ObjID].frame[PrevFrame].z;

    interPos[3]=(float) ( stream->object[ObjID].frame[NextFrame].rot1-stream->object[ObjID].frame[PrevFrame].rot1 ) * our_stepTime / MAX_stepTime;
    interPos[3]+=stream->object[ObjID].frame[PrevFrame].rot1;

    interPos[4]=(float) ( stream->object[ObjID].frame[NextFrame].rot2-stream->object[ObjID].frame[PrevFrame].rot2 ) * our_stepTime / MAX_stepTime;
    interPos[4]+=stream->object[ObjID].frame[PrevFrame].rot2;

    interPos[5]=(float) ( stream->object[ObjID].frame[NextFrame].rot3-stream->object[ObjID].frame[PrevFrame].rot3 ) * our_stepTime / MAX_stepTime;
    interPos[5]+=stream->object[ObjID].frame[PrevFrame].rot3;

    interPos[6]=(float) ( stream->object[ObjID].frame[NextFrame].rot4-stream->object[ObjID].frame[PrevFrame].rot4 ) * our_stepTime / MAX_stepTime;
    interPos[6]+=stream->object[ObjID].frame[PrevFrame].rot4;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleX -stream->object[ObjID].frame[PrevFrame].scaleX ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleX;
    *scaleX=interScale;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleY -stream->object[ObjID].frame[PrevFrame].scaleY ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleY;
    *scaleY=interScale;

    interScale = (float) ( stream->object[ObjID].frame[NextFrame].scaleZ -stream->object[ObjID].frame[PrevFrame].scaleZ ) * our_stepTime / MAX_stepTime;
    interScale += stream->object[ObjID].frame[PrevFrame].scaleZ;
    *scaleZ=interScale;


    pos[0]=interPos[0]; pos[1]=interPos[1]; pos[2]=interPos[2];
    pos[3]=interPos[3]; pos[4]=interPos[4]; pos[5]=interPos[5];
    pos[6]=interPos[6];

    #if PRINT_DEBUGGING_INFO
    fprintf(stderr,"ok \n");
    #endif

    return 1;
}






int calculateVirtualStreamPos(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAbsMilliseconds,float * pos,float * scaleX,float * scaleY,float * scaleZ)
{
   if (stream==0) { fprintf(stderr,"calculateVirtualStreamPos called with null stream\n"); return 0; }
   if (stream->object==0) { fprintf(stderr,"calculateVirtualStreamPos called with null object array\n"); return 0; }
   if (stream->numberOfObjects<=ObjID) { fprintf(stderr,"calculateVirtualStreamPos ObjID %u is out of bounds (%u)\n",ObjID,stream->numberOfObjects); return 0; }
   if (stream->object[ObjID].frame == 0 )  { fprintf(stderr,"calculateVirtualStreamPos ObjID %u does not have a frame array allocated\n",ObjID); return 0; }
   if (stream->object[ObjID].numberOfFrames == 0 ) { fprintf(stderr,"calculateVirtualStreamPos ObjID %u has 0 frames\n",ObjID); return 0; }


   if (stream->autoRefresh != 0 )
    {
         //Check for refreshed version ?
       if (stream->autoRefresh < timeAbsMilliseconds-stream->lastRefresh )
          {
            unsigned long current_size = getFileSize(stream->filename);
            if (current_size != stream->fileSize)
             {
              refreshVirtualStream(stream);
              stream->lastRefresh = timeAbsMilliseconds;
             }
          }
    }

   unsigned int FrameIDToReturn = 0;
   unsigned int FrameIDLast = 0;
   unsigned int FrameIDNext = 0;


   /*!OK , Two major cases here..! The one is a simple Next frame getter , the second is a more complicated interpolated frame getter..! */
   if ( (stream->object[ObjID].MAX_numberOfFrames == 0 ) )
   {
       fprintf(stderr,"Returning Null position for ObjID %u\n",ObjID);
       fillPosWithNull(/*stream,ObjID,*/pos,scaleX,scaleY,scaleZ);
       return 1;
   } else
   if  ( (stream->ignoreTime) || (stream->object[ObjID].MAX_numberOfFrames == 1 ) )
   {
    //We might want to ignore time and just return frame after frame on each call!
    //Also if we only got one frame for the object there is no point in trying to interpolate time etc.. so just handle things here..
    if ( stream->object[ObjID].lastFrame +1 >= stream->object[ObjID].MAX_numberOfFrames ) { stream->object[ObjID].lastFrame  = 0; }
    FrameIDToReturn = stream->object[ObjID].lastFrame;
    ++stream->object[ObjID].lastFrame;

    fillPosWithFrame(stream,ObjID,FrameIDToReturn,pos,scaleX,scaleY,scaleZ);
    fprintf(stderr,"fillPosWithFrame %u => ( %0.2f %0.2f %0.2f , %0.2f %0.2f %0.2f)\n",FrameIDToReturn,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5]);

    FrameIDLast = FrameIDToReturn;
    FrameIDNext = FrameIDToReturn+1;

    if ( FrameIDNext >= stream->object[ObjID].numberOfFrames )
     { //We 've reached the end of the stream so the last frame should truncate 0
       stream->object[ObjID].lastFrame=0;
     }

     return 1;

   } /*!END OF SIMPLE FRAME GETTER*/
   else
   { /*!START OF INTERPOLATED FRAME GETTER*/
     //fprintf(stderr,"interpolated position for ObjID %u\n",ObjID);
     //This is the case when we respect time , we will pick two frames and interpolate between them
     if ( timeAbsMilliseconds > stream->object[ObjID].MAX_timeOfFrames )
     {
       //This means we have passed the last frame.. so lets find out where we really are..
       if (stream->object[ObjID].MAX_timeOfFrames == 0 ) {
                                                           //If max time of frames is 0 then our time is also zero ( since it never goes over max )
                                                           //fprintf(stderr,"timeAbsMilliseconds can not be something more than zero");
                                                           timeAbsMilliseconds=0;
                                                         } else
                                                         { timeAbsMilliseconds = timeAbsMilliseconds % stream->object[ObjID].MAX_timeOfFrames; }
       //timeAbsMilliseconds should contain a valid value now somewhere from 0->MAX_timeOfFrames
     }

     #if PRINT_DEBUGGING_INFO
     fprintf(stderr,"Object %u has %u frames , lets search where we are \n",ObjID,stream->object[ObjID].numberOfFrames);
     #endif

     //We scan all the frames to find out the "last one" and the "next one"
     unsigned int i =0;
     for ( i=0; i <stream->object[ObjID].MAX_numberOfFrames-1; i++ )
      {
       if (( stream->object[ObjID].frame[i].time <= timeAbsMilliseconds )
                 &&
           ( timeAbsMilliseconds <= stream->object[ObjID].frame[i+1].time )  )
            {
               //This is the "next" frame!
               FrameIDLast = i;
               FrameIDNext = i+1;
               //This should be handled by raw response to zero elemetn :P
              break;
            }
      }

    //We now have our Last and Next frame , all that remains is extracting the
    //interpolated time between them..!
    return fillPosWithInterpolatedFrame(stream,ObjID,pos,scaleX,scaleY,scaleZ,FrameIDLast,FrameIDNext,timeAbsMilliseconds);

   } /*!END OF INTERPOLATED FRAME GETTER*/

    return 0;
}



int calculateVirtualStreamPosAfterTime(struct VirtualStream * stream,ObjectIDHandler ObjID,unsigned int timeAfterMilliseconds,float * pos,float * scaleX,float * scaleY, float * scaleZ)
{
   stream->object[ObjID].lastCalculationTime+=timeAfterMilliseconds;
   return calculateVirtualStreamPos(stream,ObjID,stream->object[ObjID].lastCalculationTime,pos,scaleX,scaleY,scaleZ);
}


int getVirtualStreamLastPosF(struct VirtualStream * stream,ObjectIDHandler ObjID,float * pos,float * scaleX,float * scaleY,float * scaleZ)
{
    return fillPosWithLastFrame(stream,ObjID,pos,scaleX,scaleY,scaleZ);
}





