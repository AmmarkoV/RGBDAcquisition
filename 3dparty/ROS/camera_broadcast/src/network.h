#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "AmmClient.h"

int AmmClient_ReconnectInternal(
                         struct AmmClient_Instance * instance ,
                         int triggerCheck
                       );
int AmmClient_CheckConnectionInternal(struct AmmClient_Instance * instance);


int AmmClient_CloseDeadConnectionIfNeeded(struct AmmClient_Instance * instance);

int AmmClient_RecvInternal(struct AmmClient_Instance * instance,
                   char * buffer ,
                   unsigned int * bufferSize
                  );

int AmmClient_SendInternal(struct AmmClient_Instance * instance,
                   const char * request ,
                   unsigned int requestSize,
                   int keepAlive
                  );

struct AmmClient_Instance * AmmClient_InitializeInternal(
                                                  const char * ip ,
                                                  unsigned int port ,
                                                  unsigned int socketTimeoutSeconds
                                                );

int AmmClient_CloseInternal(struct AmmClient_Instance * instance);

#endif // NETWORK_H_INCLUDED
