#include "TrajectoryParserDataStructures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


int growVirtualStreamFrames(struct VirtualObject * streamObj,unsigned int framesToAdd)
{
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
   streamObj->frame = new_frame ;
  return 1;
}


int growVirtualStreamObjectsTypes(struct VirtualStream * stream,unsigned int objectsTypesToAdd)
{
  if (objectsTypesToAdd == 0) { return 0 ; }
  if (stream == 0) { fprintf(stderr,"Given an empty stream to grow objects types on \n"); return 0 ; }
  struct ObjectType * new_objectTypes;
  new_objectTypes = (struct ObjectType *) realloc( stream->objectTypes , sizeof(struct ObjectType) * ( stream->MAX_numberOfObjectTypes+objectsTypesToAdd ));

   if (new_objectTypes == 0 )
    {
       fprintf(stderr,"Cannot add %u object types to our currently %u sized object type buffer\n",objectsTypesToAdd,stream->MAX_numberOfObjectTypes);
       return 0;
    } else
     {
      //Clean up all new object types allocated
      void * clear_from_here  = new_objectTypes+stream->MAX_numberOfObjectTypes;
      memset(clear_from_here,0,objectsTypesToAdd * sizeof(struct ObjectType));
    }

   stream->MAX_numberOfObjectTypes+=objectsTypesToAdd;
   stream->objectTypes = new_objectTypes ;
  return 1;
}



int growVirtualStreamObjects(struct VirtualStream * stream,unsigned int objectsToAdd)
{
  if (objectsToAdd == 0) { return 0 ; }
  if (stream == 0) { fprintf(stderr,"Given an empty stream to grow objects on \n"); return 0 ; }
  struct VirtualObject * new_object;
  new_object = (struct VirtualObject *) realloc( stream->object , sizeof(struct VirtualObject) * ( stream->MAX_numberOfObjects+objectsToAdd ));

   if (new_object == 0 )
    {
       fprintf(stderr,"Cannot add %u objects to our currently %u sized object buffer\n",objectsToAdd,stream->MAX_numberOfObjects);
       return 0;
    } else
    {
      //Clean up all new objects allocated
      void * clear_from_here  =  new_object+stream->MAX_numberOfObjects;
      memset(clear_from_here,0,objectsToAdd * sizeof(struct VirtualObject));
    }

   stream->MAX_numberOfObjects+=objectsToAdd;
   stream->object = new_object ;
  return 1;
}



int growVirtualStreamEvents(struct VirtualStream * stream,unsigned int eventsToAdd)
{
  if (eventsToAdd == 0) { return 0 ; }
  if (stream == 0) { fprintf(stderr,"Given an empty stream to grow objects on \n"); return 0 ; }
  struct VirtualEvent * new_event;
  new_event = (struct VirtualEvent *) realloc( stream->event , sizeof(struct VirtualEvent) * ( stream->MAX_numberOfEvents+eventsToAdd ));

   if (new_event == 0 )
    {
       fprintf(stderr,"Cannot add %u objects to our currently %u sized event buffer\n",eventsToAdd,stream->MAX_numberOfEvents);
       return 0;
    } else
    {
      //Clean up all new objects allocated
      void * clear_from_here  =  new_event+stream->MAX_numberOfEvents;
      memset(clear_from_here,0,eventsToAdd * sizeof(struct VirtualEvent));
    }

   stream->MAX_numberOfEvents+=eventsToAdd;
   stream->event = new_event ;
  return 1;
}




int growVirtualStreamConnectors(struct VirtualStream * stream,unsigned int connectorsToAdd)
{
  if (connectorsToAdd == 0) { return 0 ; }
  if (stream == 0) { fprintf(stderr,"Given an empty stream to grow objects on \n"); return 0 ; }
  struct VirtualConnector * new_connector;
  new_connector = (struct VirtualConnector *) realloc( stream->connector , sizeof(struct VirtualConnector) * ( stream->MAX_numberOfConnectors+connectorsToAdd ));

   if (new_connector == 0 )
    {
       fprintf(stderr,"Cannot add %u objects to our currently %u sized connector buffer\n",connectorsToAdd,stream->MAX_numberOfConnectors);
       return 0;
    } else
    {
      //Clean up all new objects allocated
      void * clear_from_here  =  new_connector+stream->MAX_numberOfConnectors;
      memset(clear_from_here,0,connectorsToAdd * sizeof(struct VirtualConnector));
    }

   stream->MAX_numberOfConnectors+=connectorsToAdd;
   stream->connector = new_connector ;
  return 1;
}




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


