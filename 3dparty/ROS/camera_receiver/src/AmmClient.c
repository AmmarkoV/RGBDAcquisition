#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "tools.h"
#include "network.h"
#include "protocol.h"

#include "AmmClient.h"

char * AmmClient_ReadFileToMemory(const char * filename,unsigned int *length)
{
    *length = 0;
    FILE * pFile = fopen ( filename , "rb" );

    if (pFile==0)
    {
        fprintf(stderr, "AmmServer_ReadFileToMemory failed\n");
        fprintf(stderr, "Could not read file %s \n",filename);
        return 0;
    }

    // obtain file size:
    fseek (pFile , 0 , SEEK_END);
    unsigned long lSize = ftell (pFile);
    rewind (pFile);

    // allocate memory to contain the whole file:
    unsigned long bufferSize = sizeof(char)*(lSize+1);
    char * buffer = (char*) malloc (bufferSize);
    if (buffer == 0 )
    {
        fprintf(stderr,"Could not allocate enough memory for file %s \n",filename);
        fclose(pFile);
        return 0;
    }

    // copy the file into the buffer:
    size_t result = fread (buffer,1,lSize,pFile);
    if (result != lSize)
    {
        free(buffer);
        fprintf(stderr,"Could not read the whole file onto memory %s \n",filename);
        fclose(pFile);
        return 0;
    }

    /* the whole file is now loaded in the memory buffer. */

    // terminate
    fclose (pFile);

    buffer[lSize]=0; //Null Terminate Buffer!
    *length = (unsigned int) lSize;
    return buffer;
}


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
