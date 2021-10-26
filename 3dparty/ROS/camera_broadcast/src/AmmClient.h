/** @file AmmClient.h
* @brief This is a tool than offers an efficient HTTP client with a small footprint. It is completely seperate from AmmarServer on purpose to make it as small as possible.
         Inside this repository it is mainly used by the AmmMessages mechanisms in order to provide connections back to the HTTP server for message updates
* @author Ammar Qammaz (AmmarkoV)
*/

#ifndef AMMCLIENT_H_INCLUDED
#define AMMCLIENT_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

static const char AmmClientVersion[] = "0.55";

/** @brief An instance of AmmClient, this holds the connection state and a pointer that gets automatically allocated with internals that include the sockets etc*/
struct AmmClient_Instance
{
  int clientSocket;
  int connectionOK;
  int socketOK;
  unsigned int failedReconnections;

  int socketTimeoutSeconds;

  char ip[33];
  int port;

  void * internals;
};


char * AmmClient_ReadFileToMemory(const char * filename,unsigned int *length);

/** @brief Get back a monotnic "uptime" value in the form of microseconds, a useful call to count how much time elapsed during file transfers etc*/
unsigned long AmmClient_GetTickCountMicroseconds();
/** @brief Get back a monotnic "uptime" value in the form of millitimes, a useful call to count how much time elapsed during file transfers etc*/
unsigned long AmmClient_GetTickCountMilliseconds();



/**
* @brief Check connection status for an AmmClient TCP/IP connection
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @retval 1=Connection is Ok,0=Connection Has Failed*/
int AmmClient_CheckConnection(struct AmmClient_Instance * instance);

/**
* @brief Perform a recv operation on the TCP/IP connection of this AmmClient instance
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @param Pointer to memory that will hold the output of the operation
* @param Pointer to unsigned integer that marks the size of buffer when the function is called, and contains the recv size after function exits
* @retval 1=Success Receiving,0=Failed to Receive*/
int AmmClient_Recv(
                   struct AmmClient_Instance * instance,
                   char * buffer ,
                   unsigned int * bufferSize
                  );


/**
* @brief Perform a send operation on the TCP/IP connection of this AmmClient instance
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @param Pointer to memory that holds the input buffer that will be sent across the connection
* @param Size of data to send
* @param Flag to keep the connection alive ( this should be 1 if you do multiple requests in small amounts of time to avoid reestablishing connections )
* @retval 1=Success Sending,0=Failed to Send*/
int AmmClient_Send(struct AmmClient_Instance * instance,
                   const char * request ,
                   unsigned int requestSize,
                   int keepAlive
                  );


/**
* @brief Initialize a new HTTP connection
* @ingroup AmmClient
* @param IP address of HTTP server that we want to connect to
* @param Port of HTTP server that we want to connect to
* @param Number of seconds to wait before a connection is declared dead, connections get automatically re-established but a correct value
         close to the application transmission rates here will help overall throughput
* @retval 1=Success Receiving,0=Failed to receive*/
struct AmmClient_Instance * AmmClient_Initialize(
                                                  const char * ip ,
                                                  unsigned int port,
                                                  unsigned int socketTimeoutSeconds
                                                );

/**
* @brief Stop an AmmClient instance that has been established using AmmClient_Initialize
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @retval 1=Success Closing,0=Failed to close connections gracefully*/
int AmmClient_Close(struct AmmClient_Instance * instance);


/**
* @brief Perform a GET operation and fully receive a file ( instead of crafting it yourself and doing multiple AmmClient_Send/AmmClient_Recv calls )
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @param URI of resource requested from server, a slash is prefixed so a URI containing "index.html" will result in a GET /index.html request
* @param Pointer to memory that will holds the file received from the server
* @param Pointer to unsigned integer that marks the size of buffer when the function is called, and contains the recv size after function exits
* @param Flag to keep the connection alive ( this should be 1 if you do multiple requests in small amounts of time to avoid reestablishing connections )
* @param Flag to that uses regular recv calls when set to 0 , or tries to poll the socket for maximum speed if set to 1
* @retval 1=Success Receiving,0=Failed to Receive*/
int AmmClient_RecvFile(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       char * filecontent ,
                       unsigned int* filecontentSize,
                       int keepAlive,
                       int reallyFastImplementation
                      );



/**
* @brief Perform a POST operation and fully  send a file to a server ( instead of crafting the request yourself and doing multiple AmmClient_Send/AmmClient_Recv calls )
* @ingroup AmmClient
* @param Pointer to AmmClient instance that has to have been initialized using AmmClient_Initialize
* @param URI of resource requested from server, a slash is prefixed so a URI containing "index.html" will result in a GET /index.html request
* @param Name of the POST form we are targeting
* @param Filename of file transmitted
* @param Pointer to memory that will holds the file to be sent to the server
* @param Pointer to unsigned integer that marks the size of file to be sent
* @param Flag to keep the connection alive ( this should be 1 if you do multiple requests in small amounts of time to avoid reestablishing connections )
* @retval 1=Success Receiving,0=Failed to Receive*/
int AmmClient_SendFile(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       const char * formname,
                       const char * filename ,
                       const char * contentType,
                       const char * filecontent ,
                       unsigned int filecontentSize,
                       int keepAlive
                      );

#ifdef __cplusplus
}
#endif

#endif // AMMCLIENT_H_INCLUDED
