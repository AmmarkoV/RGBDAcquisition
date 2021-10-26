#ifndef PROTOCOL_H_INCLUDED
#define PROTOCOL_H_INCLUDED

#include "AmmClient.h"

int AmmClient_SendFileInternal(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       const char * formname,
                       const char * filename ,
                       const char * contentType,
                       const char * filecontent ,
                       unsigned int filecontentSize,
                       int keepAlive
                      );


int AmmClient_RecvFileInternal(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       char * filecontent ,
                       unsigned int * filecontentSize,
                       int keepAlive,
                       int reallyFastImplementation
                      );


#endif // PROTOCOL_H_INCLUDED
