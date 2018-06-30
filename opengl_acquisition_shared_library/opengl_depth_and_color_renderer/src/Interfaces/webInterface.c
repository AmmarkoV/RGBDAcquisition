#include "webInterface.h"


#define USE_AMMARSERVER 1



#if USE_AMMARSERVER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../submodules/AmmarServer/src/AmmServerlib/AmmServerlib.h"

#include "../Scene/scene.h"

struct AmmServer_Instance  * server=0;
struct AmmServer_RH_Context cameraControl={0};
struct VirtualStream * sceneToControl = 0;


//This function prepares the content of  form context , ( content )
void * prepareCameraControlCallback(struct AmmServer_DynamicRequest  * rqst)
{
  char buf[32];

  double translation[4]={0};
  translation[3]=1.0;

  double rotation[4]={0};

  AmmServer_Warning("New Camera State");
  //==============================================================
  if ( _GETcpy(rqst,"x",buf,32) )  { translation[0] = atof(buf); }
  if ( _GETcpy(rqst,"y",buf,32) )  { translation[1] = atof(buf); }
  if ( _GETcpy(rqst,"z",buf,32) )  { translation[2] = atof(buf); }
  //===========================================================
  if ( _GETcpy(rqst,"qX",buf,32) ) { rotation[0] = atof(buf); }
  if ( _GETcpy(rqst,"qY",buf,32) ) { rotation[1] = atof(buf); }
  if ( _GETcpy(rqst,"qZ",buf,32) ) { rotation[2] = atof(buf); }
  if ( _GETcpy(rqst,"qW",buf,32) ) { rotation[3] = atof(buf); }
  //===========================================================

  if (sceneToControl!=0)
  {
    sceneSetOpenGLExtrinsicCalibration(sceneToControl,rotation,translation,1000.0);
    setupSceneCameraBeforeRendering(sceneToControl);
  }


  strncpy(rqst->content,"<html><body>OK</body></html>",rqst->MAXcontentSize);
  rqst->contentSize=strlen(rqst->content);
  return 0;
}





void init_dynamic_content()
{
  AmmServer_AddResourceHandler(server,&cameraControl,"/control.html",4096,0,&prepareCameraControlCallback,SAME_PAGE_FOR_ALL_CLIENTS);
}



int initializeWebInterface(int argc, char *argv[],struct VirtualStream * scene)
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
      sceneToControl=scene;
      return 1;
    }


 return 0;
}
#else

int initializeWebInterface(int argc, char *argv[],struct VirtualStream * scene)
{
  return 0;
}

#endif // USE_AMMARSERVER

