#include "TrajectoryParserDataStructures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

