/*
AmmarServer , simple template executable

URLs: http://ammar.gr
Written by Ammar Qammaz a.k.a. AmmarkoV 2012

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../../../AmmarServer/src/AmmServerlib/AmmServerlib.h"
#include "../tools/Codecs/codecs.h"


#define DEFAULT_BINDING_PORT 9090

unsigned int maxUploadFileSizeAllowedMB=4; /*MB*/

char webserver_root[MAX_FILE_PATH]="src/Services/MyLoader/res/"; // <- change this to the directory that contains your content if you dont want to use the default public_html dir..
char uploads_root[MAX_FILE_PATH]="uploads/";
char templates_root[MAX_FILE_PATH]="public_html/templates/";

unsigned int uploadsFilesSize=0;
unsigned int uploadsDataSize=0;
//The decleration of some dynamic content resources..
struct AmmServer_Instance  * default_server=0;
struct AmmServer_RH_Context uploadProcessor={0};
struct AmmServer_RH_Context indexProcessor={0};




//This function prepares the content of  stats context , ( stats.content )
void * prepare_index_callback(struct AmmServer_DynamicRequest  * rqst)
{
  snprintf(rqst->content,rqst->MAXcontentSize,"<html><body>\
  <form enctype=\"multipart/form-data\" action=\"upload.html\" method=\"POST\">\
       <input type=\"hidden\" name=\"rawresponse\" value=\"NO\" />\
       File to upload: <input name=\"uploadedfile\" type=\"file\" />\
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
       <input type=\"submit\" value=\"Upload File\" name=\"submit\" />&nbsp;&nbsp;\
   </form></body></html>\
   ");
  rqst->contentSize=strlen(rqst->content);
  return 0;
}




void * processUploadCallback(struct AmmServer_DynamicRequest  * rqst)
{
   char uploadedFileUNSANITIZEDPath[513]={0};
   AmmServer_POSTNameOfFile (default_server,rqst,0,uploadedFileUNSANITIZEDPath,512);
   AmmServer_Warning("Unsanitized filename is %s \n",uploadedFileUNSANITIZEDPath);

   if (AmmServer_StringHasSafePath(uploads_root,uploadedFileUNSANITIZEDPath))
   {
    char * uploadedFilePath = uploadedFileUNSANITIZEDPath;
    char finalPath[2049]={0};
    //snprintf(finalPath,2048,"%s/%s/%s-%s",webserver_root,uploads_root,storeID,uploadedFilePath);
    AmmServer_POSTArgToFile (default_server,rqst,0,finalPath);

    unsigned int filePointerLength=0;
    char * data=AmmServer_POSTArgGetPointer(default_server,rqst,0,&filePointerLength);
   }
  return 0;
}



//This function adds a Resource Handler for the pages stats.html and formtest.html and associates stats , form and their callback functions
void init_dynamic_content()
{
  AmmServer_SetIntSettingValue(default_server,AMMSET_MAX_POST_TRANSACTION_SIZE,(maxUploadFileSizeAllowedMB+1)*1024*1024); //+1MB for headers etc..

  AmmServer_AddResourceHandler(default_server,&uploadProcessor,"/upload.html",webserver_root,4096,0,&processUploadCallback,DIFFERENT_PAGE_FOR_EACH_CLIENT|ENABLE_RECEIVING_FILES);
  AmmServer_DoNOTCacheResourceHandler(default_server,&uploadProcessor);

  AmmServer_AddResourceHandler(default_server,&indexProcessor,"/index.html",webserver_root,4096,0,&prepare_index_callback,DIFFERENT_PAGE_FOR_EACH_CLIENT);

}

//This function destroys all Resource Handlers and free's all allocated memory..!
void close_dynamic_content()
{
    AmmServer_RemoveResourceHandler(default_server,&uploadProcessor,1);
    AmmServer_RemoveResourceHandler(default_server,&indexProcessor,1);
}




int main(int argc, char *argv[])
{
    printf("\nAmmar Server %s starting up..\n",AmmServer_Version());
    //Check binary and header spec
    AmmServer_CheckIfHeaderBinaryAreTheSame(AMMAR_SERVER_HTTP_HEADER_SPEC);
    //Register termination signal for when we receive SIGKILL etc
    AmmServer_RegisterTerminationSignal(&close_dynamic_content);

    char bindIP[MAX_IP_STRING_SIZE];
    strncpy(bindIP,"0.0.0.0",MAX_IP_STRING_SIZE);

    unsigned int port=DEFAULT_BINDING_PORT;

    //Kick start AmmarServer , bind the ports , create the threads and get things going..!
    default_server = AmmServer_StartWithArgs(
                                             "myloader",
                                              argc,argv , //The internal server will use the arguments to change settings
                                              //If you don't want this look at the AmmServer_Start call
                                              bindIP,
                                              port,
                                              0, /*This means we don't want a specific configuration file*/
                                              webserver_root,
                                              templates_root
                                              );


    if (!default_server) { AmmServer_Error("Could not start server , shutting down everything.."); exit(1); }

    //Create dynamic content allocations and associate context to the correct files
    init_dynamic_content();
    //stats.html and formtest.html should be availiable from now on..!

         while ( (AmmServer_Running(default_server))  )
           {
             //Main thread should just sleep and let the background threads do the hard work..!
             //In other applications the programmer could use the main thread to do anything he likes..
             //The only caveat is that he would takeup more CPU time from the server and that he would have to poll
             //the AmmServer_Running() call once in a while to make sure everything is in order
             //usleep(60000);
             sleep(1);
           }

    //Delete dynamic content allocations and remove stats.html and formtest.html from the server
    close_dynamic_content();

    //Stop the server and clean state
    AmmServer_Stop(default_server);
    AmmServer_Warning("Ammar Server stopped\n");
    return 0;
}
