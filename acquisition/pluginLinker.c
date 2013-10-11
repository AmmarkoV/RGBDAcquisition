#include "pluginLinker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <dlfcn.h>


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


char * pluginStrings[] = {   "dummyName" , "dummyPath "                                 , " dummyPath.so"               ,
                             //----------------------------------------------------------------------------
                             "V4L2"      ,"../v4l2_acquisition_shared_library/"        , "libV4L2Acquisition.so"       ,
                             "V4L2Stereo","../v4l2stereo_acquisition_shared_library/"  , "libV4L2StereoAcquisition.so" ,
                             "Freenect"  ,"../libfreenect_acquisition_shared_library/" , "libFreenectAcquisition.so"   ,
                             "OpenNI1"   ,"../openni1_acquisition_shared_library/"     ,"libOpenNI1Acquisition.so"     ,
                             "OpenNI2"   ,"../openni2_acquisition_shared_library/"     , "libOpenNI2Acquisition.so"    ,
                             "OpenGL"    ,"../opengl_acquisition_shared_library/"      , "libOpenGLAcquisition.so"     ,
                             "Template"  ,"../template_acquisition_shared_library/"    , "libTemplateAcquisition.so"   ,
                             "Network"   ,"../network_acquisition_shared_library/"    , "libNetworkAcquisition.so"     ,
                             //----------------------------------------------------------------------------
                             "Dont Erase the following!! they serve as a working warning if a new plugin is introduced but not specified here"  ,
                             "Dont Forget to add name for new plugin here "  ,
                             "Dont Forget to add paths for new plugins here "  ,
                             "Dont Forget to add library names for new plugins here "
                        };


char * getPluginStr(int moduleID,int strID)
{
    if (3 * moduleID + strID > NUMBER_OF_POSSIBLE_MODULES *3  )
    {
       fprintf(stderr,"getPluginStr(%u,%u) cannot return a result , plugin not declared correctly\n",moduleID,strID);
    }

    fprintf(stderr,"getPluginStr(%u,%u) returning %s\n",moduleID,strID,pluginStrings[  3 * moduleID + strID ]);
    return pluginStrings[  3 * moduleID + strID ];
}


struct acquisitionPluginInterface plugins[NUMBER_OF_POSSIBLE_MODULES]={0};

void * remoteNetworkDLhandle=0;
int (*startPushingToRemoteNetwork) (char * , int);
int (*stopPushingToRemoteNetwork) (int);
int (*pushImageToRemoteNetwork) (int,int,void *,unsigned int,unsigned int,unsigned int,unsigned int);






int getPluginPath(char * possiblePath, char * libName , char * pathOut, unsigned int pathOutLength)
{


   char* ldPreloadPath;
   ldPreloadPath= getenv("LD_PRELOAD");
   if (ldPreloadPath!=0) { fprintf(stderr,"Todo Implement check in paths : `%s` \n",ldPreloadPath); }


   char pathTester[2048]={0};


   if (getcwd(pathTester, sizeof(pathTester)) != 0)
         fprintf(stdout, "Current working dir: %s\n", pathTester);


   sprintf(pathTester,"%s/%s",possiblePath,libName);
   if (fileExists(pathTester))   {
                                   fprintf(stderr,"Found plugin %s at Path %s\n",libName,possiblePath);
                                   strncpy(pathOut,pathTester,pathOutLength);
                                   return 1;
                                 } else
   if (fileExists(libName))      {
                                   fprintf(stderr,"Found plugin %s at CurrentDir\n",libName);
                                   //strncpy(pathOut,libName,pathOutLength);

                                   strcpy(pathOut,"./"); //<-TODO CHECK BOUNDS HERE ETC..
                                   strcat(pathOut,libName);

                                   return 1;
                                 }



   //TODO HANDLE LIBRARY PATH STRINGS
   // They look like /opt/ros/groovy/lib:/usr/local/cuda-4.2/cuda/lib64:/usr/local/cuda-4.2/cuda/lib
   char* ldPath;
   ldPath= getenv("LD_LIBRARY_PATH");
   if (ldPath!=0)        { fprintf(stderr,"Todo Implement check in paths : `%s` \n",ldPath);  }

   return 0;
}






int linkToNetworkTransmission(char * moduleName,char * modulePossiblePath ,char * moduleLib)
{
   char *error;
   char functionNameStr[1024]={0};

   if (!getPluginPath(modulePossiblePath,moduleLib,functionNameStr,1024))
       {
          fprintf(stderr,RED "Could not find %s (try adding it to current directory)\n" NORMAL , moduleLib);
          return 0;
       }

   remoteNetworkDLhandle = dlopen (functionNameStr, RTLD_LAZY);
   if (!remoteNetworkDLhandle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, moduleName , functionNameStr , dlerror());
        return 0;
       }

  dlerror();    /* Clear any existing error */


  //Start Stop ================================================================================================================
  startPushingToRemoteNetwork = dlsym(remoteNetworkDLhandle, "networkBackbone_startPushingToRemote" );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW "Could not find a definition of networkBackbone_startPushingToRemote : %s\n" NORMAL,error); }

  stopPushingToRemoteNetwork = dlsym(remoteNetworkDLhandle, "networkBackbone_stopPushingToRemote" );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW "Could not find a definition of networkBackbone_stopPushingToRemote : %s\n" NORMAL,error); }

  pushImageToRemoteNetwork = dlsym(remoteNetworkDLhandle, "networkBackbone_pushImageToRemote" );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW "Could not find a definition of networkBackbone_pushImageToRemote : %s\n" NORMAL,error); }

  return 1;
}



int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID)
{

   char *error;
   char functionNameStr[1024]={0};

   if (!getPluginPath(modulePossiblePath,moduleLib,functionNameStr,1024))
       {
          fprintf(stderr,RED "Could not find %s (try adding it to current directory)\n" NORMAL , moduleLib);
          return 0;
       }

   plugins[moduleID].handle = dlopen (functionNameStr, RTLD_LAZY);
   if (!plugins[moduleID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, moduleName , functionNameStr , dlerror());
        return 0;
       }

    dlerror();    /* Clear any existing error */


  //Start Stop ================================================================================================================
  sprintf(functionNameStr,"start%sModule",moduleName);
  plugins[moduleID].startModule = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }

  sprintf(functionNameStr,"stop%sModule",moduleName);
  plugins[moduleID].stopModule = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }

  //================================================================================================================
  sprintf(functionNameStr,"map%sDepthToRGB",moduleName);
  plugins[moduleID].mapDepthToRGB = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"map%sRGBToDepth",moduleName);
  plugins[moduleID].mapRGBToDepth = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }


  sprintf(functionNameStr,"create%sDevice",moduleName);
  plugins[moduleID].createDevice = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"destroy%sDevice",moduleName);
  plugins[moduleID].destroyDevice = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }


  sprintf(functionNameStr,"get%sNumberOfDevices",moduleName);
  plugins[moduleID].getNumberOfDevices = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }


  sprintf(functionNameStr,"seek%sFrame",moduleName);
  plugins[moduleID].seekFrame = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"snap%sFrames",moduleName);
  plugins[moduleID].snapFrames = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }


  sprintf(functionNameStr,"getLast%sColorTimestamp",moduleName);
  plugins[moduleID].getLastColorTimestamp = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"getLast%sDepthTimestamp",moduleName);
  plugins[moduleID].getLastDepthTimestamp = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }


  sprintf(functionNameStr,"get%sColorWidth",moduleName);
  plugins[moduleID].getColorWidth = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorHeight",moduleName);
  plugins[moduleID].getColorHeight = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorDataSize",moduleName);
  plugins[moduleID].getColorDataSize = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorChannels",moduleName);
  plugins[moduleID].getColorChannels = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorBitsPerPixel",moduleName);
  plugins[moduleID].getColorBitsPerPixel = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorPixels",moduleName);
  plugins[moduleID].getColorPixels = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorFocalLength",moduleName);
  plugins[moduleID].getColorFocalLength = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorPixelSize",moduleName);
  plugins[moduleID].getColorPixelSize = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sColorCalibration",moduleName);
  plugins[moduleID].getColorCalibration = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"set%sColorCalibration",moduleName);
  plugins[moduleID].setColorCalibration = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }



  sprintf(functionNameStr,"get%sDepthWidth",moduleName);
  plugins[moduleID].getDepthWidth = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthHeight",moduleName);
  plugins[moduleID].getDepthHeight = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthDataSize",moduleName);
  plugins[moduleID].getDepthDataSize = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthChannels",moduleName);
  plugins[moduleID].getDepthChannels = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthBitsPerPixel",moduleName);
  plugins[moduleID].getDepthBitsPerPixel = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthPixels",moduleName);
  plugins[moduleID].getDepthPixels = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthFocalLength",moduleName);
  plugins[moduleID].getDepthFocalLength = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthPixelSize",moduleName);
  plugins[moduleID].getDepthPixelSize = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"get%sDepthCalibration",moduleName);
  plugins[moduleID].getDepthCalibration = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
  sprintf(functionNameStr,"set%sDepthCalibration",moduleName);
  plugins[moduleID].setDepthCalibration = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)  { fprintf (stderr, YELLOW  "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }

  return 1;
}

int unlinkPlugin(ModuleIdentifier moduleID)
{
    if (plugins[moduleID].handle==0) { return 1; }
    dlclose(plugins[moduleID].handle);
    plugins[moduleID].handle=0;
    return 1;
}
