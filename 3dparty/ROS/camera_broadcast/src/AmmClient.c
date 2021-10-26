#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "tools.h"
#include "network.h"
#include "protocol.h"

#include "AmmClient.h"

unsigned long AmmClient_GetTickCountMicroseconds()
{
    return AmmClient_GetTickCountMicrosecondsInternal();
}

unsigned long AmmClient_GetTickCountMilliseconds()
{
    return AmmClient_GetTickCountMillisecondsInternal();
}


int AmmClient_Reconnect(
    struct AmmClient_Instance * instance,
    int triggerCheck
)
{
    return AmmClient_ReconnectInternal( instance, triggerCheck );
}

int AmmClient_CheckConnection(struct AmmClient_Instance * instance)
{
    return AmmClient_CheckConnectionInternal(instance);
}

int AmmClient_Recv(struct AmmClient_Instance * instance,
                   char * buffer,
                   unsigned int * bufferSize
                  )
{
    return AmmClient_RecvInternal( instance,  buffer,  bufferSize );
}


int AmmClient_RecvFile(
    struct AmmClient_Instance * instance,
    const char * URI,
    char * filecontent,
    unsigned int * filecontentSize,
    int keepAlive,
    int reallyFastImplementation
)
{
    return AmmClient_RecvFileInternal(
               instance,
               URI,
               filecontent,
               filecontentSize,
               keepAlive,
               reallyFastImplementation
           );
}


int AmmClient_Send(struct AmmClient_Instance * instance,
                   const char * request,
                   unsigned int requestSize,
                   int keepAlive
                  )
{
    return AmmClient_SendInternal( instance, request, requestSize, keepAlive );
}

int AmmClient_SendFile(
    struct AmmClient_Instance * instance,
    const char * URI,
    const char * formname,
    const char * filename,
    const char * contentType,
    const char * filecontent,
    unsigned int filecontentSize,
    int keepAlive
)
{
    return AmmClient_SendFileInternal(
               instance,
               URI,
               formname,
               filename,
               contentType,
               filecontent,
               filecontentSize,
               keepAlive
           );
}

struct AmmClient_Instance * AmmClient_Initialize(
    const char * ip,
    unsigned int port,
    unsigned int socketTimeoutSeconds
)
{
    return AmmClient_InitializeInternal( ip, port,socketTimeoutSeconds );
}


int AmmClient_Close(struct AmmClient_Instance * instance)
{
    return AmmClient_CloseInternal(instance);
}
