#include "webInterface.h"

#define USE_AMMARSERVER 1
#if USE_AMMARSERVER
#include "../../submodules/AmmarServer/src/AmmServerlib/AmmServerlib.h"

struct AmmServer_Instance  * server=0;
struct AmmServer_RH_Context cameraControl={0};



//This function prepares the content of  form context , ( content )
void * prepareCameraControlCallback(struct AmmServer_DynamicRequest  * rqst)
{
  char buf[32];
  float x=0.0,y=0.0,z=0.0,qX=0.0,qY=0.0,qZ=0.0,qW=1.0;

  AmmServer_Warning("New Camera State");
  //==================================================
  if ( _GETcpy(rqst,"x",buf,32) )  { x = atof(buf); }
  if ( _GETcpy(rqst,"y",buf,32) )  { y = atof(buf); }
  if ( _GETcpy(rqst,"z",buf,32) )  { z = atof(buf); }
  //==================================================
  if ( _GETcpy(rqst,"qX",buf,32) ) { qX = atof(buf); }
  if ( _GETcpy(rqst,"qY",buf,32) ) { qY = atof(buf); }
  if ( _GETcpy(rqst,"qZ",buf,32) ) { qZ = atof(buf); }
  if ( _GETcpy(rqst,"qW",buf,32) ) { qW = atof(buf); }
  //==================================================


  strncpy(rqst->content,"<html><body>OK</body></html>",rqst->MAXcontentSize);
  rqst->contentSize=strlen(rqst->content);
  return 0;
}





void init_dynamic_content()
{
//  AmmServer_AddResourceHandler(server,&cameraControl,"/control.html",4096,0,&prepare_stats_content_callback,SAME_PAGE_FOR_ALL_CLIENTS);
}



int initializeWebInterface(int argc, char *argv[])
{
     server = AmmServer_StartWithArgs(
                                             "opengl",
                                              argc,argv , //The internal server will use the arguments to change settings
                                              //If you don't want this look at the AmmServer_Start call
                                              "0.0.0.0",
                                              8080,
                                              0, /*This means we don't want a specific configuration file*/
                                              "public_html/",
                                              "public_html/"
                                              );
    if (server)
    {
      init_dynamic_content();
      return 1;
    }


 return 0;
}
#else

int initializeWebInterface(int argc, char *argv[])
{
  return 0;
}

#endif // USE_AMMARSERVER

