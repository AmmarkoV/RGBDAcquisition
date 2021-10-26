
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>

#include <poll.h>

#include "AmmClient.h"
#include "protocol.h"
#include "network.h"



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */



#define MICROSECONDS_TO_WAIT_FOR_RESPONSE 2000 //2 msec round trip time should be enough , ultimately this only matters in reallyFastImplementation since the regular will wait until the recv is complete

#define BUFFERSIZE 8192

int AmmClientTransmitBoundary(struct AmmClient_Instance * instance,const char * boundary,int keepAlive,int first,int after,int last)
{
  char transmission[256]={"------------------------8a7a8e26954c7dff"};
  int transmissionSize = 0;

  if (first)
  {
    transmissionSize = snprintf(transmission,256,"--%s\r\n",boundary);
  } else
  if (after)
  {
    transmissionSize = snprintf(transmission,256,"\r\n--%s\r\n",boundary);
  } else
  if (last)
  {
    transmissionSize = snprintf(transmission,256,"\r\n--%s--\r\n",boundary);
  } else
  {
    transmissionSize = snprintf(transmission,256,"\r\n--%s\r\n",boundary);
  }

  //fprintf(stderr,"%s",transmission);

return AmmClient_SendInternal(instance,transmission,transmissionSize,keepAlive);

}


//https://www.w3schools.com/php/php_file_upload.asp
//nc -l 0.0.0.0 8080
//curl -F "submit=1" -F "fileToUpload=@/home/dji/catkin_ws/src/camera_broadcast/src/image.jpg" ammar.gr/stream/upload.php
//curl -F "fileToUpload=@/home/ammar/Documents/Programming/AmmarServer/src/AmmClient/test.jpg" -F "submit=1"  ammar.gr/stream/upload.php
int AmmClient_SendFileInternal(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       const char * formname,
                       const char * filename ,
                       const char * contentType,
                       const char * filecontent ,
                       unsigned int filecontentSize,
                       int keepAlive
                      )
{
  //Ok first of all we want to generate a random boundary..
  char boundary[128]={0};
  unsigned int rpA=rand()%9999;
  unsigned int rpB=rand()%9999;
  unsigned int rpC=rand()%9999;
  unsigned int rpD=rand()%9999;
  unsigned int boundarySize = snprintf(boundary,128,"------------------------%04u%04u%04u%04u",rpA,rpB,rpC,rpD);
  //--------------------------------------------------------------

  //Our transmission consists of 4 parts
  // 1st HEADER
  // 2nd FILE PREAMBLE
  // 3d  BINARY FILA
  // 4th POST FILE STUFF

  int success = 0;
  int steps   = 0;

  //We prepare them before sending the file to be able to know the content length of the file
  unsigned int bufferBeforeSize=0;
  char bufferBeforeFile[BUFFERSIZE+1]={0};
  bufferBeforeSize = snprintf(bufferBeforeFile,BUFFERSIZE,"Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n",formname,filename,contentType);

  unsigned int bufferAfterSize=0;
  char bufferAfterFile[BUFFERSIZE+1]={0};
  bufferAfterSize = snprintf(bufferAfterFile,BUFFERSIZE,"Content-Disposition: form-data; name=\"submit\"\r\n\r\n1");


  unsigned int contentLength = bufferBeforeSize + boundarySize + 4 + filecontentSize + boundarySize + 6 + bufferAfterSize + boundarySize + 8;


  char header[BUFFERSIZE+1]={0};
  //Send the header Connection: keep-alive\r\nTransfer-Encoding: chunked\r\n
  int headerSize = snprintf(header,BUFFERSIZE,"POST %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: AmmClient/%s\r\nAccept: */*\r\nContent-Length: %u\r\nContent-Type: multipart/form-data; boundary=%s\r\n\r\n",URI,instance->ip,AmmClientVersion,contentLength,boundary);

  //Send everything..
  ++steps; success+=AmmClient_SendInternal(instance,header,headerSize,keepAlive);
  ++steps; success+=AmmClientTransmitBoundary(instance,boundary,keepAlive,1,0,0);
  ++steps; success+=AmmClient_SendInternal(instance,bufferBeforeFile,bufferBeforeSize,keepAlive);
  ++steps; success+=AmmClient_SendInternal(instance,filecontent,filecontentSize,keepAlive);
  ++steps; success+=AmmClientTransmitBoundary(instance,boundary,keepAlive,0,1,0);
  ++steps; success+=AmmClient_SendInternal(instance,bufferAfterFile,bufferAfterSize,keepAlive);
  ++steps; success+=AmmClientTransmitBoundary(instance,boundary,keepAlive,0,0,1);

 return ( success==steps );
}





int isFileDownloadComplete(const char * content,unsigned int contentSize)
{
 //fprintf(stderr," Content (%u bytes) : %s \n" ,contentSize , content);
  char * contentLengthStr=strstr(content,"Content-length:");
  if (contentLengthStr!=0)
  {
      contentLengthStr+=16;
      unsigned int contentLength = atoi(contentLengthStr);
      //fprintf(stderr,YELLOW " Content Length = %u  \n" NORMAL , contentLength );
      if (contentLength!=0)
      {
        unsigned int headerSize = contentLengthStr - content;

        if (headerSize+contentLength <= contentSize)
              {
                //fprintf(stderr,GREEN " File fully downloaded ( %u header + %u content = %u we have )\n" NORMAL , headerSize , contentLength , contentSize);
                return 1;
              }
      }
  }
  //fprintf(stderr,RED " File not fully downloaded\n" NORMAL );
 return 0;
}



int AmmClient_RecvFileInternalClean(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       char * filecontent ,
                       unsigned int * filecontentSize,
                       int keepAlive
                      )
{
 snprintf(filecontent,*filecontentSize,"GET /%s HTTP/1.1\nConnection: keep-alive\n\n",URI);

 unsigned int bytesReceived=0;

 if (AmmClient_Send(instance,filecontent,strlen(filecontent),keepAlive))
 {
     memset(filecontent,0,strlen(filecontent));
     unsigned int doneReceiving=0;

     usleep(MICROSECONDS_TO_WAIT_FOR_RESPONSE);

     while (!doneReceiving)
     {
       /* Upon successful completion, recv() shall return the length of the message in bytes.
        If no messages are available to be received and the peer has performed an orderly shutdown,
        recv() shall return 0.*/
       int result = recv(instance->clientSocket, filecontent+bytesReceived, *filecontentSize-bytesReceived, 0);
       if (result == 0 ) {
                            fprintf(stderr,".");
                            doneReceiving=1;
                         } else
       if (result < 0 )  {
                           fprintf(stderr,RED "Failed to Recv error : %u\n" NORMAL,errno);
                           AmmClient_CloseDeadConnectionIfNeeded(instance);
                           return 0;
                         } else
                         {
                           bytesReceived+=result;
                           //fprintf(stderr,GREEN" Received %u more bytes ( total %u ) \n" NORMAL , result , bytesReceived);
                         }

       if (bytesReceived>=*filecontentSize) { doneReceiving=1;}
       if (isFileDownloadComplete(filecontent,bytesReceived))
                         {  *filecontentSize = bytesReceived; return 1; }

     }
 }

  *filecontentSize = bytesReceived;

  AmmClient_CloseDeadConnectionIfNeeded(instance);
  AmmClient_CheckConnectionInternal(instance);

 return 0;
}




int AmmClient_RecvFileInternalFast(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       char * filecontent ,
                       unsigned int * filecontentSize,
                       int keepAlive
                      )
{
 snprintf(filecontent,*filecontentSize,"GET /%s HTTP/1.1\nConnection: keep-alive\n\n",URI);

 unsigned int bytesReceived=0;

 if (AmmClient_Send(instance,filecontent,strlen(filecontent),keepAlive))
 {
     memset(filecontent,0,strlen(filecontent));
     unsigned int doneReceiving=0;
     unsigned int connectionHalted=0;
     int result =0;

     struct pollfd fd;

     usleep(MICROSECONDS_TO_WAIT_FOR_RESPONSE);


     while (!doneReceiving)
     {
      int recvCalled=0;

      //Do ultra fast polling to check if connection has data..
      //---------------------------------------------------------------------
      fd.fd = instance->clientSocket; // your socket handler
      fd.events = POLLIN;
      int ret = poll(&fd, 1, 1); // 1ms for timeout
      switch (ret) {
                     case -1:
                      doneReceiving=1;
                     break;
                     case 0:
                        fprintf(stderr,".");
                        ++connectionHalted;

                        result=0;
                        doneReceiving=1;
                     break;
                     default:
                       result=recv(instance->clientSocket, filecontent+bytesReceived, *filecontentSize-bytesReceived, 0);
                       recvCalled=1;
                     break;
                   }
      //---------------------------------------------------------------------


       if (recvCalled)
       {
        if (result == 0 ) {
                            fprintf(stderr,"!");
                            ++connectionHalted;
                            doneReceiving=1;
                          } else
        if (result < 0 )  {
                           fprintf(stderr,RED "\nFailed to Recv error : %u\n" NORMAL,errno);
                           AmmClient_CloseDeadConnectionIfNeeded(instance);
                           return 0;
                          } else
                          {
                           bytesReceived+=result;
                           //fprintf(stderr,GREEN" Received %u more bytes ( total %u ) \n" NORMAL , result , bytesReceived);
                          }

        if (bytesReceived>=*filecontentSize) { doneReceiving=1;}

        if (isFileDownloadComplete(filecontent,bytesReceived))
          {
            *filecontentSize = bytesReceived;
            return 1;
          }

       }
     }
 }

  *filecontentSize = bytesReceived;

  //AmmClient_CloseDeadConnectionIfNeeded(instance);
  //AmmClient_CheckConnectionInternal(instance);

 return 0;
}



int AmmClient_RecvFileInternal(
                       struct AmmClient_Instance * instance,
                       const char * URI ,
                       char * filecontent ,
                       unsigned int * filecontentSize,
                       int keepAlive ,
                       int reallyFastImplementation
                      )
{
 if (reallyFastImplementation)
 {
 //If we are here it means we are in a terrible hurry
 //The "fast" implementation uses poll to check sockets
 //and will not block, providing a much faster implementation..
 return AmmClient_RecvFileInternalFast(
                                        instance,
                                        URI ,
                                        filecontent ,
                                        filecontentSize,
                                        keepAlive
                                       );

 }
 else
 {
 //This is the clean and simple version for receiving files
 //it uses regular recv and works @ 25Hz due to system blocking
 //at recv for a predetermined amount of time.
 return AmmClient_RecvFileInternalClean(
                                        instance,
                                        URI ,
                                        filecontent ,
                                        filecontentSize,
                                        keepAlive
                                       );
 }

 return 0;
}
