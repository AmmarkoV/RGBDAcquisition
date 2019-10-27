#include "TrajectoryParserDataStructures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "TrajectoryCalculator.h"

#include "../ModelLoader/model_loader_hardcoded.h"
#include "../ModelLoader/model_loader_tri.h"
#include "../ModelLoader/model_loader_transform_joints.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

#define USE_HASHMAPS 0

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



int dummy_strcasecmp_internal(const char * input1,const char * input2)
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

ObjectIDHandler getObjectID(struct VirtualStream * stream,const char * name, unsigned int * found)
{
  if (found==0) { fprintf(stderr,"Can't get object id without a valid value to return if found\n"); return 0; }
  *found=0;

  if (name==0) { fprintf(stderr,"Can't get object id without a valid name\n"); return 0; }
  if (stream==0) { fprintf(stderr,"Can't get object id (%s) for un allocated stream\n",name); }
  if (stream->object==0) { fprintf(stderr,"Can't get object id (%s) for un allocated object array\n",name); }


  if (stream->MAX_numberOfObjects<=stream->numberOfObjects)
    {
     fprintf(stderr,"number of objects is larger than maximum objects ????? %u/%u objects \n",stream->numberOfObjects,stream->MAX_numberOfObjects);
     return 0;
    }

  if (stream->debug)
    { fprintf(stderr,"Searching %s among %u/%u objects \n",name,stream->numberOfObjects , stream->MAX_numberOfObjects); }


 #if USE_HASHMAPS
 unsigned long index;
 *found=(hashMap_GetULongPayload(stream->objectHash,name,&index)!=0);
 return (unsigned int) index;
 #else
  unsigned int i=0;
  for (i=0; i<stream->numberOfObjects; i++ )
   {
     if (stream->object[i].name!=0)
     {
       if (dummy_strcasecmp_internal(name,stream->object[i].name)==0)
         {
          if (stream->debug)
               { fprintf(stderr,"Found it @ stream->object[%u]\n",i); }
              *found=1;
              return i;
         }
     }
   }

   return 0;
 #endif // USE_HASHMAPS

}

ObjectTypeID getObjectTypeID(struct VirtualStream * stream,const char * typeName,unsigned int * found)
{
  if (stream==0) { fprintf(stderr,"Can't get object id (%s) for un allocated stream\n",typeName); }
  if (stream->objectTypes==0) { fprintf(stderr,"Can't get object id (%s) for un allocated object type array\n",typeName); }

 #if USE_HASHMAPS
 unsigned long index;
 *found=(hashMap_GetULongPayload(stream->objectTypesHash,typeName,&index)!=0);
 return (unsigned int) index;
 #else
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
 #endif // USE_HASHMAPS
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



struct JointState * allocateEnoughJointSpaceForStateOfObjectID(
                                                                struct VirtualStream * stream ,
                                                                unsigned int ObjID
                                                               )
{
 unsigned int numberOfBones=stream->objectTypes[stream->object[ObjID].type].numberOfBones;

 if (numberOfBones!=0)
 {
 if (ObjID==0)
    { /*Camera does not have joints*/ } else
    { fprintf(stderr,"allocateEnoughJointSpaceForStateOfObjectID for objid %u has %u bones ..\n",ObjID,numberOfBones); }


   fprintf(stderr,"Also allocating %u joints for this model..\n",numberOfBones);
   struct JointState * js = ( struct JointState * ) malloc(sizeof(struct JointState));
   if (js!=0)
   {
    unsigned int sizeJointTimesNumberOfBones = numberOfBones * sizeof(struct Joint);
    js->joint = ( struct Joint *) malloc(sizeJointTimesNumberOfBones);
    if (js->joint!=0)
     {
       memset(js->joint,0,sizeJointTimesNumberOfBones);
       js->numberOfJoints = numberOfBones;
     } else { fprintf(stderr,"Could not allocate joint space for all joints of objID %u ", ObjID ); }
   } else { fprintf(stderr,"Could not allocate joint space for objID %u ", ObjID ); }
   return js;
 }

 return 0;
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
  //Todo check timeMilliseconds is our new state actually a replacement for an old one ?
  if (stream->object[ObjID].MAX_numberOfFrames<=stream->object[ObjID].numberOfFrames+1) { growVirtualStreamFrames(&stream->object[ObjID],FRAMES_TO_ADD_STEP); }
  //Now we should definately have enough space for our new frame
  if (stream->object[ObjID].MAX_numberOfFrames<=stream->object[ObjID].numberOfFrames+1) { fprintf(stderr,"Cannot add new POS instruction to Object %u \n",ObjID); return 0; }

  //We have the space so lets fill our new frame spot ..!
  unsigned int pos = stream->object[ObjID].numberOfFrames;
  ++stream->object[ObjID].numberOfFrames;

  // 1 is object name
  stream->object[ObjID].frame[pos].time = timeMilliseconds;
  stream->object[ObjID].frame[pos].isQuaternion = 0;
  stream->object[ObjID].frame[pos].isEulerRotation = 0;

  stream->object[ObjID].frame[pos].scaleX = scaleX;
  stream->object[ObjID].frame[pos].scaleY = scaleY;
  stream->object[ObjID].frame[pos].scaleZ = scaleZ;
  stream->object[ObjID].frame[pos].R = R;
  stream->object[ObjID].frame[pos].G = G;
  stream->object[ObjID].frame[pos].B = B;
  stream->object[ObjID].frame[pos].Alpha = Alpha;

  stream->object[ObjID].frame[pos].jointList=allocateEnoughJointSpaceForStateOfObjectID( stream , ObjID );
  stream->object[ObjID].frame[pos].hasNonDefaultJointList = 0; // Initially not changed..!

  if (coordLength > 0 ) {  stream->object[ObjID].frame[pos].x = coord[0]; }
  if (coordLength > 1 ) {  stream->object[ObjID].frame[pos].y = coord[1]; }
  if (coordLength > 2 ) {  stream->object[ObjID].frame[pos].z = coord[2]; }

  if (coordLength > 3 ) {  stream->object[ObjID].frame[pos].rot1 = coord[3];     }
  if (coordLength > 4 ) {  stream->object[ObjID].frame[pos].rot2 = coord[4];     }
  if (coordLength > 5 ) {  stream->object[ObjID].frame[pos].rot3 = coord[5];     }
  if (coordLength > 6 ) {  stream->object[ObjID].frame[pos].rot4 = coord[6];     }

  if (coordLength==6)   {  stream->object[ObjID].frame[pos].isEulerRotation = 1; } else
  if (coordLength==7)   {  stream->object[ObjID].frame[pos].isQuaternion = 1;    } else
                        {  fprintf(stderr,"addStateToObjectID: ObjID=%u frame[%u] incorrect rotation component..\n",ObjID,pos); }

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
                              const char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength ,
                              float scaleX , float scaleY ,float scaleZ ,
                              float R , float G , float B , float Alpha
                       )
{
 //fprintf(stderr,"addStateToObject\n");
 unsigned int ObjFound = 0;
 unsigned int ObjID = getObjectID(stream,name,&ObjFound);
 if (ObjFound)
  {
     return addStateToObjectID(stream,ObjID,timeMilliseconds,coord,coordLength,scaleX,scaleY,scaleZ,R,G,B,Alpha);
  }
  fprintf(stderr,"Could not Find object %s \n",name);
  return 0;
}


int addStateToObjectMini(
                              struct VirtualStream * stream ,
                              char * name  ,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength
                       )
{
 unsigned int ObjFound = 0;
 unsigned int ObjID = getObjectID(stream,name,&ObjFound);
 //fprintf(stderr,"addStateToObjectMini(%s,id=%u,found=%u)\n",name,ObjID,ObjFound);
 if (ObjFound)
  {
     return addStateToObjectID(
                               stream,ObjID,timeMilliseconds,
                               coord,coordLength,
                               stream->object[ObjID].scaleX,
                               stream->object[ObjID].scaleY,
                               stream->object[ObjID].scaleZ,
                               stream->object[ObjID].R,
                               stream->object[ObjID].G,
                               stream->object[ObjID].B,
                               stream->object[ObjID].Transparency
                              );
  }
  fprintf(stderr,"addStateToObjectMini: Could not Find object `%s`\n",name);
  return 0;
}




int getRotationOrderFromString(const char * rotationOrderStr)
{
 unsigned int rotationOrder=0;

 if (strcmp(rotationOrderStr,"xyz")==0) { rotationOrder=ROTATION_ORDER_XYZ; } else
 if (strcmp(rotationOrderStr,"xzy")==0) { rotationOrder=ROTATION_ORDER_XZY; } else
 if (strcmp(rotationOrderStr,"yxz")==0) { rotationOrder=ROTATION_ORDER_YXZ; } else
 if (strcmp(rotationOrderStr,"yzx")==0) { rotationOrder=ROTATION_ORDER_YZX; } else
 if (strcmp(rotationOrderStr,"zxy")==0) { rotationOrder=ROTATION_ORDER_ZXY; } else
 if (strcmp(rotationOrderStr,"zyx")==0) { rotationOrder=ROTATION_ORDER_ZYX; } else
 if (strcmp(rotationOrderStr,"rpy")==0) { rotationOrder=ROTATION_ORDER_RPY; }

 return rotationOrder;
}


int changeObjectRotationOrder(
                                  struct VirtualStream * stream ,
                                  struct ModelList * modelStorage,
                                  char * name  ,
                                  char * rotationOrderStr
                                )
{
 unsigned int objFound = 0;
 unsigned int objTypeFound = 0;
 unsigned int objID = getObjectID(stream,name,&objFound);


 if(modelStorage!=0)
 {
 if (objFound)
 {
  unsigned int objTypeID = getObjectTypeID(stream,stream->object[objID].typeStr,&objTypeFound );
  if (objTypeFound)
  {
    unsigned int modelID = stream->objectTypes[objTypeID].modelListArrayNumber;
    struct Model *mod = &modelStorage->models[modelID];
    mod->rotationOrder=getRotationOrderFromString(rotationOrderStr);
    fprintf(stderr,"Setting rotation order %u for model %s \n",mod->rotationOrder,mod->pathOfModel);
    return 1;
  } else  { fprintf(stderr,"changeModelRotationOrder: Could not find object type..\n"); }
 } else { fprintf(stderr,"changeModelRotationOrder: Could not find object..\n"); }
 } else { fprintf(stderr,"changeModelRotationOrder: No model storage allocated\n"); }


 return 0;
}















int changeModelJointRotationOrder(
                                  struct VirtualStream * stream ,
                                  struct ModelList * modelStorage,
                                  char * name  ,
                                  char * jointName,
                                  char * modelOrder
                                )
{
 unsigned int boneIDResult;
 unsigned int objFound = 0;
 unsigned int objTypeFound = 0;
 unsigned int objID = getObjectID(stream,name,&objFound);


 if(modelStorage!=0)
 {
 if (objFound)
 {
  unsigned int objTypeID = getObjectTypeID(stream,stream->object[objID].typeStr,&objTypeFound );
  if (objTypeFound)
  {
   unsigned int modelID = stream->objectTypes[objTypeID].modelListArrayNumber;
   struct Model *mod = &modelStorage->models[modelID];
   if (mod->type==TRI_MODEL)
   {
    struct TRI_Model * triM = (struct TRI_Model * ) mod->modelInternalData;

    if ( findTRIBoneWithName(triM,jointName,&boneIDResult) )
    {
     unsigned int rotationOrder=getRotationOrderFromString(modelOrder);

     if (
         setTRIJointRotationOrder(
                                  triM,
                                  boneIDResult,
                                  rotationOrder
                                 )
        )
     {

        return 1;

     } else { fprintf(stderr,"changeModelRotationOrder: could not find a correct joint ..\n"); }
    } else { fprintf(stderr,"changeModelRotationOrder: could not find joint..\n"); }
   } else { fprintf(stderr,"changeModelRotationOrder: cannot change model rotation order on non TRI models, (objid=%u / objtype=%u / mod->type=%u) ..\n",objID,objTypeID,mod->type); }
  } else  { fprintf(stderr,"changeModelRotationOrder: Could not find object type..\n"); }
 } else { fprintf(stderr,"changeModelRotationOrder: Could not find object..\n"); }
 } else { fprintf(stderr,"changeModelRotationOrder: No model storage allocated\n"); }


 return 0;
}


int changeAllPosesInObjectState(
                                struct VirtualStream * stream ,
                                struct ModelList * modelStorage,
                                const char * name  ,
                                const char * jointName,
                                unsigned int timeMilliseconds ,
                                float * coord ,
                                unsigned int coordLength
                               )
{
 if (stream==0)                                     {   fprintf(stderr,"Invalid stream \n"); return 0; }
 if (modelStorage==0)                               {   fprintf(stderr,"Invalid model storage \n"); return 0; }
 if ( (name==0)||(jointName==0)||(coord==0) )       {   fprintf(stderr,"Invalid values to add as a pose \n"); return 0; }

 int foundExactTimestamp=0;
 unsigned int ObjFound = 0;
 //fprintf(stderr,"Adding pose to object %s \n",name);
 unsigned int ObjID = getObjectID(stream,name,&ObjFound);

 //fprintf(stderr,"Object Found = %u , Object ID = %u \n",ObjFound,ObjID);
 if (ObjFound)
  {
    unsigned int pos=0;
    if ( stream->object[ObjID].frame[pos].jointList !=0 )
    {
       //pos = getExactStreamPosFromTimestamp(stream,ObjID,timeMilliseconds,&foundExactTimestamp); 
       if(stream->object[ObjID].numberOfFrames>0)
       {
        for (pos=0; pos<stream->object[ObjID].numberOfFrames; pos++)
        {
           
        
        unsigned int objectTypeID = stream->object[ObjID].type;

        unsigned int modelID = stream->objectTypes[objectTypeID].modelListArrayNumber;
        //fprintf(stderr,"Accessing model %u/%u\n", modelID,modelStorage->currentNumberOfModels);
        if (modelID<modelStorage->currentNumberOfModels)
        {
        struct Model * mod = (struct Model *) &modelStorage->models[modelID];
        if (mod!=0)
        {
        int boneFound=0;

        //fprintf(stderr,"Set mod->initialized=%u\n", mod->initialized);

        unsigned int boneID = getModelBoneIDFromBoneName(mod,jointName,&boneFound);

        if (boneFound)
        {
           stream->object[ObjID].frame[pos].hasNonDefaultJointList = 1;  //Whatever we set it is now set..!
           stream->object[ObjID].frame[pos].jointList->numberOfJoints = mod->numberOfBones;

           stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useEulerRotation=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useQuaternion=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useMatrix4x4=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=0;

           if (coordLength==3)
           {
            if (stream->debug)
                { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms euler angle (%0.2f %0.2f %0.2f) \n",ObjID,pos,boneID,timeMilliseconds,coord[0],coord[1],coord[2]); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useEulerRotation=1;

            //By default the euler rotation order will be ZYX but this can be changed using the POSE_ROTATION_ORDER command
            stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=getModelBoneRotationOrderFromBoneName(mod,boneID);

            if (stream->debug)
                { fprintf(stderr,"bone %u => rotation order %u \n",boneID,stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder); }

            if (stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder==0)
            {
              stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=ROTATION_ORDER_ZYX;
            }

            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot1=coord[0];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot2=coord[1];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot3=coord[2];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
           } else
           if (coordLength==4)
           {
            if (stream->debug)
                { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms quaternion(%0.2f %0.2f %0.2f %0.2f) \n",ObjID,pos,boneID,timeMilliseconds,coord[0],coord[1],coord[2],coord[3]); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useQuaternion=1;
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot1=coord[0];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot2=coord[1];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot3=coord[2];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot4=coord[3];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
           } else
           if (coordLength==16)
           {
            if (stream->debug)
                 { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms Matrix4x4 \n",ObjID,pos,boneID,timeMilliseconds); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useMatrix4x4=1;
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
            unsigned int z=0;
            for (z=0; z<16; z++)
              { stream->object[ObjID].frame[pos].jointList->joint[boneID].m[z]=coord[z]; }
           } else
           {
             fprintf(stderr,RED "Unknown coordinate length ( %u )  obj=%u pos=%u bone=%u @ %u ms Matrix4x4 \n" NORMAL,coordLength,ObjID,pos,boneID,timeMilliseconds);
           }

           return 1;
        } else { fprintf(stderr,"Could not find exact bone %u for %s \n",boneID,name); }
        } else { fprintf(stderr,"Could not find data of model %u for %s \n",modelID,name); }
        } else { fprintf(stderr,"Could not find exact model %u for %s \n",modelID,name); }
        } 
       } else  { fprintf(stderr,"Could not find any timestamp for %s \n", name); }
    } else     { fprintf(stderr,"Could not Find a joint list for %s ( model not loaded? )\n",name); }
  } else       { fprintf(stderr,"Could not Find object %s \n",name); }

  return 0;
}









int addPoseToObjectState(
                              struct VirtualStream * stream ,
                              struct ModelList * modelStorage,
                              const char * name  ,
                              const char * jointName,
                              unsigned int timeMilliseconds ,
                              float * coord ,
                              unsigned int coordLength
                        )
{
 if (stream==0)                                     {   fprintf(stderr,"Invalid stream \n"); return 0; }
 if (modelStorage==0)                               {   fprintf(stderr,"Invalid model storage \n"); return 0; }
 if ( (name==0)||(jointName==0)||(coord==0) )       {   fprintf(stderr,"Invalid values to add as a pose \n"); return 0; }

 int foundExactTimestamp=0;
 unsigned int ObjFound = 0;
 //fprintf(stderr,"Adding pose to object %s \n",name);
 unsigned int ObjID = getObjectID(stream,name,&ObjFound);

 //fprintf(stderr,"Object Found = %u , Object ID = %u \n",ObjFound,ObjID);
 if (ObjFound)
  {
    unsigned int pos=0;
    if ( stream->object[ObjID].frame[pos].jointList !=0 )
    {
       pos = getExactStreamPosFromTimestamp(stream,ObjID,timeMilliseconds,&foundExactTimestamp);

       if(foundExactTimestamp)
       {
        unsigned int objectTypeID = stream->object[ObjID].type;

        unsigned int modelID = stream->objectTypes[objectTypeID].modelListArrayNumber;
        //fprintf(stderr,"Accessing model %u/%u\n", modelID,modelStorage->currentNumberOfModels);
        if (modelID<modelStorage->currentNumberOfModels)
        {
        struct Model * mod = (struct Model *) &modelStorage->models[modelID];
        if (mod!=0)
        {
        int boneFound=0;

        //fprintf(stderr,"Set mod->initialized=%u\n", mod->initialized);

        unsigned int boneID = getModelBoneIDFromBoneName(mod,jointName,&boneFound);

        if (boneFound)
        {
           stream->object[ObjID].frame[pos].hasNonDefaultJointList = 1;  //Whatever we set it is now set..!
           stream->object[ObjID].frame[pos].jointList->numberOfJoints = mod->numberOfBones;

           stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useEulerRotation=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useQuaternion=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].useMatrix4x4=0;
           stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=0;

           if (coordLength==3)
           {
            if (stream->debug)
                { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms euler angle (%0.2f %0.2f %0.2f) \n",ObjID,pos,boneID,timeMilliseconds,coord[0],coord[1],coord[2]); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useEulerRotation=1;

            //By default the euler rotation order will be ZYX but this can be changed using the POSE_ROTATION_ORDER command
            stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=getModelBoneRotationOrderFromBoneName(mod,boneID);

            if (stream->debug)
                { fprintf(stderr,"bone %u => rotation order %u \n",boneID,stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder); }

            if (stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder==0)
            {
              stream->object[ObjID].frame[pos].jointList->joint[boneID].eulerRotationOrder=ROTATION_ORDER_ZYX;
            }

            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot1=coord[0];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot2=coord[1];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot3=coord[2];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
           } else
           if (coordLength==4)
           {
            if (stream->debug)
                { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms quaternion(%0.2f %0.2f %0.2f %0.2f) \n",ObjID,pos,boneID,timeMilliseconds,coord[0],coord[1],coord[2],coord[3]); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useQuaternion=1;
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot1=coord[0];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot2=coord[1];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot3=coord[2];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].rot4=coord[3];
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
           } else
           if (coordLength==16)
           {
            if (stream->debug)
                 { fprintf(stderr,"Set obj=%u pos=%u bone=%u @ %u ms Matrix4x4 \n",ObjID,pos,boneID,timeMilliseconds); }
            stream->object[ObjID].frame[pos].jointList->joint[boneID].useMatrix4x4=1;
            stream->object[ObjID].frame[pos].jointList->joint[boneID].altered=1;
            unsigned int z=0;
            for (z=0; z<16; z++)
              { stream->object[ObjID].frame[pos].jointList->joint[boneID].m[z]=coord[z]; }
           } else
           {
             fprintf(stderr,RED "Unknown coordinate length ( %u )  obj=%u pos=%u bone=%u @ %u ms Matrix4x4 \n" NORMAL,coordLength,ObjID,pos,boneID,timeMilliseconds);
           }

           return 1;
        } else { fprintf(stderr,"Could not find exact bone %u for %s \n",boneID,name); }
        } else { fprintf(stderr,"Could not find data of model %u for %s \n",modelID,name); }
        } else { fprintf(stderr,"Could not find exact model %u for %s \n",modelID,name); }
       } else  { fprintf(stderr,"Could not find exact timestamp %u for %s \n",timeMilliseconds, name); }
    } else     { fprintf(stderr,"Could not Find a joint list for %s ( model not loaded? )\n",name); }
  } else       { fprintf(stderr,"Could not Find object %s \n",name); }

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

   //fprintf(stderr,"addConnectorToVirtualStream (%0.2f , %0.2f , %0.2f ) \n",stream->connector[pos].R , stream->connector[pos].G,  stream->connector[pos].B);

   stream->connector[pos].Transparency = (float) Alpha/100;
   stream->connector[pos].scale = scale;

   ++stream->numberOfConnectors;
   return 1;
}


int getObjectVirtualStreamPositionAtIndex(struct VirtualStream * stream,ObjectIDHandler objID,unsigned int frameIndex,float * coords)
{
  if (stream==0) { return 0; }
  if (stream->object==0) { return 0; }
  if (stream->object[objID].frame==0) { return 0; }

  coords[0] = stream->object[objID].frame[frameIndex].x;
  coords[1] = stream->object[objID].frame[frameIndex].y;
  coords[2] = stream->object[objID].frame[frameIndex].z;
  //-----------
  coords[3] = stream->object[objID].frame[frameIndex].rot1;
  coords[4] = stream->object[objID].frame[frameIndex].rot2;
  coords[5] = stream->object[objID].frame[frameIndex].rot3;
  coords[6] = stream->object[objID].frame[frameIndex].rot4;

 return 1;
}


int generateAngleObjectsForVirtualStream(struct VirtualStream * stream, struct ModelList * modelStorage,char * excludeObjectType)
{
  char name[512]={0};

  float offset = 0.5;
  unsigned int found=0;
  unsigned int frame=0;
  unsigned int duration=10000;

  float scale=0.03; // 0.025
  addObjectTypeToVirtualStream(stream,"autoGenAngleObject","sphere",0);


  float coords[7]={100,100,100,0,0,0,0};
  unsigned int i=0;
  unsigned int originalNumberOfObjects = stream->numberOfObjects;
  //i=0 is the camera ! we dont include it
  for (i=1; i<originalNumberOfObjects; i++)
  {


    if ( strcmp(excludeObjectType,stream->object[i].typeStr)==0 )
    {
      fprintf(stderr,"excluding objid %u ( objtype %s [ %s ] ) from generateAngleObjectsForVirtualStream \n",i,stream->object[i].typeStr,excludeObjectType);
    } else
    {
       unsigned int planetObj = i;

        snprintf(name,512,"objAngleForObj%u",i);

        getObjectVirtualStreamPositionAtIndex(stream,planetObj,0,coords);
        coords[1]+=offset;
        //No Rotation for Sattelites ( they are spheres anyway )
        coords[3]=0.0; coords[4]=0.0; coords[5]=0.0; coords[6]=0.0;

        addObjectToVirtualStream(
                                 stream ,
                                 modelStorage,
                                 name , "autoGenAngleObject" ,
                                 255,0,255,0, /**/0,
                                 coords,7,
                                 scale,
                                 scale,
                                 scale,
                                 planetObj
                                );


       unsigned int z;
       for (z=0; z<stream->object[planetObj].numberOfFrames; z++)
       {
        getObjectVirtualStreamPositionAtIndex(stream,planetObj,z,coords);
        coords[1]+=offset;
        //No Rotation for Sattelites ( they are spheres anyway )
        coords[3]=0.0; coords[4]=0.0; coords[5]=0.0; coords[6]=0.0;
        addStateToObject(stream,name,stream->object[planetObj].frame[z].time ,coords,7,scale,scale,scale,255,0,255,0 );
       }

       unsigned int satteliteObj = getObjectID(stream,name,&found);


       if (! affixSatteliteToPlanetFromFrameForLength(stream,satteliteObj,planetObj,frame,duration) )
               {
                fprintf(stderr,RED "Could not affix Object %u to Object %u for %u frames ( starting @ %u )\n" NORMAL , satteliteObj,planetObj,duration,frame);
               }
    }
  }
 return 1;
}





//For now this is written without range checks..
int splitRawFilenameToDirectoryFilenameAndExtension(
                                                     const char * inputFilename,
                                                     char * directory ,
                                                     char * filename ,
                                                     char * extension ,
                                                     char * filenameWExtension,
                                                     unsigned int outputSizes
                                                    )
{
   //Basic case
   strcpy(directory,"./");         //strcpy also copies null terminator
   strcpy(filename,inputFilename); //strcpy also copies null terminator
   extension[0]=0;
   unsigned int inputFilenameLength=strlen(inputFilename);
   unsigned int extensionStart = inputFilenameLength , filenameStart = inputFilenameLength  ,filenameSpan = 0,  directorySpan = 0; //directoryStart = inputFilenameLength ,


   if (inputFilenameLength==0) { return 0; }
   unsigned int i=inputFilenameLength-1;

   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   //We first handle the extension
   while ((i>0)&&(inputFilename[i]!='.')) { --i; }
   if (i==0)
     {
       /* could not find the content type.. */
       //fprintf(stderr,"Could not find extension..\n");
       i=inputFilenameLength-1;
     } else
   if (i+1>=inputFilenameLength)
     {
       /* found the dot at i BUT it is the last character so no extension is possible..! */
       i=inputFilenameLength-1;
     } else
     {
       //we found a legit extension..!
       extensionStart = i+1;
       const char * startOfExtension = &inputFilename[i+1]; // do not include . ( dot )
       strcpy(extension,startOfExtension);
     }
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -



   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   //We then handle the directory/filename division
   while ((i>0)&&(inputFilename[i]!='/')) { --i; }

   filenameStart  = i;
   filenameSpan = extensionStart - filenameStart;
   const char * startOfFilename = &inputFilename[i+1]; // do not include . ( dot )
   strncpy(filename,startOfFilename,filenameSpan);
   filename[filenameSpan]=0;
   if (filenameSpan>2) { filename[filenameSpan-2]=0; }

   snprintf(filenameWExtension,outputSizes,"%s.%s",filename,extension);

   if (i==0) {
               //This whole thing is a filename
               //If we could not find a directory it means the resulting string is all just a big filename

               strcpy(filename,inputFilename); //fix bug if no directory is present.. , thisi s badly written
               //fprintf(stderr,"This whole thing was just a filename all along (%s/%s) ..\n",filename,inputFilename);
               return 1;
             } //<- could not find the content type..
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // - - - - - - - - - - - - - - - - - - - - - - - - - - -


  //If we reached this place we have a directory(!)

   //directoryStart  = i;
   const char * startOfDirectory = &inputFilename[0]; // do not include . ( dot )
   directorySpan = filenameStart;
   strncpy(directory,startOfDirectory,directorySpan);
   directory[directorySpan]=0;

  return 1;
}



int loadObjectTypeModelForVirtualStream(
                                        struct VirtualStream * stream ,
                                        const char * modelName ,
                                        unsigned int objTypeID
                                       )
{
  fprintf(stderr,"Loading Model with name %s and objTypeID  %u \n",modelName,objTypeID);

   char directory[MAX_MODEL_PATHS]={0};
   char filename[MAX_MODEL_PATHS]={0};
   char extension[MAX_MODEL_PATHS]={0};
   char filenameWExtension[MAX_MODEL_PATHS]={0};
   splitRawFilenameToDirectoryFilenameAndExtension(
                                                     modelName,
                                                     directory ,
                                                     filename ,
                                                     extension ,
                                                     filenameWExtension ,
                                                     MAX_MODEL_PATHS
                                                    );

  fprintf(stderr,"Loading Model with name %s ( dir = %s , file = %s , ext = %s )  and objTypeID  %u \n",modelName,directory,filename,extension,objTypeID);


   if (
       loadModelToModelList(
                            stream->associatedModelList ,
                            directory ,
                            filename ,
                            extension ,
                            &stream->objectTypes[objTypeID].modelListArrayNumber //This gets back the correct model to draw
                           )

       )
   {
    stream->objectTypes[objTypeID].numberOfBones =  getModelListBoneNumber( stream->associatedModelList , stream->objectTypes[objTypeID].modelListArrayNumber);
    fprintf(stderr,GREEN "loadObjectTypeModelForVirtualStream succeeded\n" NORMAL);
    return 1;
   } else
   { fprintf(stderr,RED "loadObjectTypeModelForVirtualStream failed\n" NORMAL);            }
 return 0;
}




int addObjectToVirtualStream(
                              struct VirtualStream * stream ,
                              struct ModelList * modelStorage,
                              const char * name , const char * type ,
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

   #if USE_HASHMAPS
   hashMap_AddULong(stream->objectHash,name,pos);
   #endif // USE_HASHMAPS

   fprintf(stderr,GREEN "ObjectID %u has name %s and type %s \n" NORMAL,pos,name,type);
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
  // return found;
}





int addSimpleObject(
                    struct VirtualStream * stream ,
                    const char * type ,
                    unsigned char R, unsigned char G , unsigned char B ,
                    float * coords ,
                    float scale
                   )
{
  char autoName[512];
  snprintf(autoName,512,"autoObject%u",stream->numberOfObjects);

  return
  addObjectToVirtualStream(
                              stream ,
                              stream->modelStorage,
                              autoName, type ,
                              R,G,B,0,
                              0,
                              coords,
                              6,
                              scale,
                              scale,
                              scale,
                              0
                            );
}








int removeObjectFromVirtualStream(struct VirtualStream * stream , unsigned int ObjID )
{
 fprintf(stderr,"removeObjectFromVirtualStream is a stub , it is not implemented , ObjID %u stayed in stream (%p) \n",ObjID,stream);
 return 0;
}


int modelFileExists(const char * filename)
{
 if (filename==0) { return 0; }
 //fprintf(stderr,"Checking if file (%s) exists : ",filename);
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */
            fclose(fp);
            //fprintf(stderr,"yes\n");
            return 1;
          }
 /* doesnt exist */
 //fprintf(stderr,"no\n");
 return 0;
}


int downloadModel(const char * model , const char * path)
{
    fprintf(stderr,YELLOW "downloadModel `%s` `%s`\n" NORMAL,model,path);

  unsigned int itIsAHardcodedModel;
  isModelnameAHardcodedModel(model,&itIsAHardcodedModel);

  if (itIsAHardcodedModel) {return 1;}
  if (modelFileExists(model)) { return 1; } else
  {
    fprintf(stderr,YELLOW "We don't have the model so we will try to download it from `%s`\n" NORMAL,path);

    char runScript[1024]={0};
    snprintf(runScript,1024,"Models/downloadModel.sh \"%s\"",path);
    int i=system(runScript);
    if (i==0)
       {
          return 1;
       }
  }
  return 0;
}



int addObjectTypeToVirtualStream(
                                 struct VirtualStream * stream ,
                                 const char * type ,
                                 const char * model,
                                 const char * webLink
                                )
{
    if (stream->MAX_numberOfObjectTypes<=stream->numberOfObjectTypes+1) { growVirtualStreamObjectsTypes(stream,OBJECT_TYPES_TO_ADD_STEP); }
    //Now we should definately have enough space for our new frame
    if (stream->MAX_numberOfObjectTypes<=stream->numberOfObjectTypes+1) { fprintf(stderr,"Cannot add new OBJECTTYPE instruction\n"); }

    //We have the space so lets fill our new object spot ..!
    unsigned int pos = stream->numberOfObjectTypes;


     if (downloadModel(model,webLink))
      {
       #if USE_HASHMAPS
         hashMap_AddULong(stream->objectTypesHash,type,pos);
       #endif // USE_HASHMAPS

       strcpy(stream->objectTypes[pos].name,type);
       strcpy(stream->objectTypes[pos].model,model);

       fprintf(stderr,"addedObjectType(%s,%s) with ID %u , now to load model \n",type,model,pos);
       if (
           loadObjectTypeModelForVirtualStream(
                                               stream ,
                                               model ,
                                               pos
                                             )
          )
          {
            ++stream->numberOfObjectTypes;
            return 1;
          }
      } else
      { fprintf(stderr,"Could not find model\n"); }

    return 0;
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

    ++stream->numberOfEvents;

    return 1; // <- we a
}




void myStrCpy(char * destination,const char * source,unsigned int maxDestinationSize)
{
  unsigned int i=0;
  while ( (i<maxDestinationSize) && (source[i]!=0) ) { destination[i]=source[i]; ++i; }
}
