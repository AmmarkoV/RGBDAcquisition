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
                             "DepthSense"   ,"../depthsense_acquisition_shared_library/"    , "libDepthSenseAcquisition.so"     ,
                             "Realsense"   ,"../librealsense_acquisition_shared_library/"    , "libRealsenseAcquisition.so"     ,
                             "Desktop"   ,"../desktop_acquisition_shared_library/"           , "libDesktopAcquisition.so"     ,
                             "Scripted"   ,"../scripted_acquisition_shared_library/"         , "dummy.so"     ,
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

    //fprintf(stderr,"getPluginStr(%u,%u) returning %s\n",moduleID,strID,pluginStrings[  3 * moduleID + strID ]);
    return pluginStrings[  3 * moduleID + strID ];
}


struct acquisitionPluginInterface plugins[NUMBER_OF_POSSIBLE_MODULES]={0};

void * remoteNetworkDLhandle=0;
int (*startPushingToRemoteNetwork) (char * , int , unsigned int , unsigned int );
int (*stopPushingToRemoteNetwork) (int);
int (*pushImageToRemoteNetwork) (int,int,void *,unsigned int,unsigned int,unsigned int,unsigned int);




int getPluginPathFromEnvVariable(const char * envVar ,const char * libName , char * pathOut, unsigned int pathOutLength )
{
   // They look like /opt/ros/groovy/lib:/usr/local/cuda-4.2/cuda/lib64:/usr/local/cuda-4.2/cuda/lib
   char pathTester[4097]={0};

   char* ldPath;
   ldPath= getenv(envVar);
   if (ldPath!=0)
     {
       //fprintf(stderr,"Todo Implement check in paths : `%s` \n",ldPath);
       char * directoriesToCheck = (char*) malloc(sizeof(char) * (strlen(ldPath)+1) );
       if (directoriesToCheck==0)
       {
         fprintf(stderr,"Could not allocate a small chunk of memory to formulate strings for Environment check\n");
         return 0;
       }

       char * endOfPath=directoriesToCheck;
       char * startOfPath=directoriesToCheck;
       if (directoriesToCheck!=0)
       {
        strncpy(directoriesToCheck,ldPath,strlen(ldPath));
        directoriesToCheck[strlen(ldPath)]=0; //Ensure null termination

        while (endOfPath!=0)
          {
           endOfPath = strstr(startOfPath,":");
           if (endOfPath!=0)
           {
             *endOfPath=0;


              //========================================================
                 snprintf(pathTester,4096,"%s/%s",startOfPath,libName);
              //   fprintf(stderr,"Check @ %s\n",pathTester);
              //========================================================
                 if (acquisitionFileExists(pathTester))
                                 {
                                     fprintf(stderr,"Found plugin %s at %s/%s\n",libName,startOfPath,libName);
                                     snprintf(pathOut,pathOutLength,GREEN "%s/%s" NORMAL,startOfPath,libName);
                                     free(directoriesToCheck);
                                     return 1;
                                 }
              //========================================================



             *endOfPath=':'; // Restore string to its prior state..
             startOfPath=endOfPath+1;
           }
          }
          //========================================================
            snprintf(pathTester,4096,"%s/%s",startOfPath,libName);
          //  fprintf(stderr,"Last Check @ %s\n",pathTester);
          //========================================================
           if (acquisitionFileExists(pathTester))
                                 {
                                     fprintf(stderr,GREEN "Found plugin %s at %s/%s\n" NORMAL,libName,startOfPath,libName);
                                     snprintf(pathOut,pathOutLength,"%s/%s",startOfPath,libName);
                                     free(directoriesToCheck);
                                     return 1;
                                 }
          //========================================================

        free(directoriesToCheck);
       }
     }
 return 0;
}



int getPluginPath(char * possiblePath, char * libName , char * pathOut, unsigned int pathOutLength)
{
   //fprintf(stderr,GREEN "Now searching for plugin %s \n" NORMAL,libName);

   //LD_PRELOAD proeceeds our lookups..!
   if (getPluginPathFromEnvVariable("LD_PRELOAD",libName,pathOut,pathOutLength))              { return 1; }

   //First of all let's check current directory
   if (acquisitionFileExists(libName))
                                 {
                                   fprintf(stderr,GREEN "Found plugin %s at current directory \n" NORMAL,libName);
                                   snprintf(pathOut,pathOutLength,"./%s",libName);
                                   return 1;
                                 }

   char pathTester[4097]={0};
   snprintf(pathTester,4096,"./%s",libName);
   if (acquisitionFileExists(libName))
                                 {
                                   fprintf(stderr,GREEN "Found plugin %s at current directory but by using a trick\n" NORMAL,libName);
                                   snprintf(pathOut,pathOutLength,"./%s",libName);
                                   return 1;
                                 }


   snprintf(pathTester,4096,"%s/%s",possiblePath,libName);
   if (acquisitionFileExists(pathTester))
                                 {
                                   fprintf(stderr,GREEN "Found plugin %s at predictable subpath %s\n" NORMAL,libName,possiblePath);
                                   snprintf(pathOut,pathOutLength,"%s/%s",possiblePath,libName);
                                   return 1;
                                 }




   if (getPluginPathFromEnvVariable("LD_LIBRARY_PATH",libName,pathOut,pathOutLength))
     {
       fprintf(stderr,GREEN "Found plugin %s at LD_LIBRARY_PATH\n" NORMAL,libName);
       return 1;
     }

   if (getPluginPathFromEnvVariable("RGBD_ACQUISITION",libName,pathOut,pathOutLength))
     {
       fprintf(stderr,YELLOW "Found plugin at obsolete old environent variable , consider exporting RGBDACQUISITION_REDIST\n" NORMAL);
       return 1;
     }

   if (getPluginPathFromEnvVariable("RGBDACQUISITION_PATH",libName,pathOut,pathOutLength))
    {
      fprintf(stderr,YELLOW "Found plugin at root dir environent variable which is weird , consider exporting RGBDACQUISITION_REDIST \n" NORMAL);
      return 1;
    }
   if (getPluginPathFromEnvVariable("RGBDACQUISITION_REDIST",libName,pathOut,pathOutLength))
    {
      fprintf(stderr,GREEN "Found plugin with RGBDACQUISITION_REDIST environment var\n" NORMAL);
      return 1;
    }




   fprintf(stderr,YELLOW "Could not find plugin library %s..\n " NORMAL,libName);
   if (getcwd(pathTester, 4096) != 0)
         fprintf(stdout, "  working dir was : %s\n", pathTester);

   return 0;
}






int linkToNetworkTransmission(char * moduleName,char * modulePossiblePath ,char * moduleLib)
{
   char *error;
   char functionNameStr[1025]={0};

   if (!getPluginPath(modulePossiblePath,moduleLib,functionNameStr,1024))
       {
          fprintf(stderr,RED "Could not find netlib %s (try adding it to current directory)\n" NORMAL , moduleLib);
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


int isPluginLoaded(ModuleIdentifier moduleID)
{
  if (!plugins[moduleID].handle)
       {
         return 0;
       }
   return 1;
}

void * linkFunction(ModuleIdentifier moduleID,char * functionName,char * moduleName)
{
  char *error;
  char functionNameStr[1025]={0};
  sprintf(functionNameStr,functionName,moduleName);
  void * linkPtr = dlsym(plugins[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)
     { fprintf (stderr, YELLOW "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
     //else { fprintf (stderr, GREEN "Found %s \n" NORMAL ,functionNameStr ); }

  return linkPtr;
}



int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID)
{
   char functionNameStr[1025]={0};

   if (!getPluginPath(modulePossiblePath,moduleLib,functionNameStr,1024))
       {
          fprintf(stderr,RED "Could not find %s (try adding it to current directory)\n" NORMAL , moduleLib);
          return 0;
       }

   plugins[moduleID].handle = dlopen (functionNameStr, RTLD_LAZY);
   if (!plugins[moduleID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for `%s` plugin from `%s`\n Error : %s\n" NORMAL, moduleName , functionNameStr , dlerror());
        return 0;
       }

    dlerror();    /* Clear any existing error */


  //Start Stop ================================================================================================================


  plugins[moduleID].getModuleCapabilities = linkFunction(moduleID,"get%sCapabilities",moduleName);

  plugins[moduleID].startModule = linkFunction(moduleID,"start%sModule",moduleName);
  plugins[moduleID].stopModule = linkFunction(moduleID,"stop%sModule",moduleName);
  plugins[moduleID].mapDepthToRGB = linkFunction(moduleID,"map%sDepthToRGB",moduleName);
  plugins[moduleID].mapRGBToDepth = linkFunction(moduleID,"map%sRGBToDepth",moduleName);
  plugins[moduleID].listDevices = linkFunction(moduleID,"list%sDevices",moduleName);

  plugins[moduleID].changeResolution = linkFunction(moduleID,"change%sResolution",moduleName);
  plugins[moduleID].createDevice = linkFunction(moduleID,"create%sDevice",moduleName);
  plugins[moduleID].destroyDevice  = linkFunction(moduleID,"destroy%sDevice",moduleName);

  plugins[moduleID].enableStream  = linkFunction(moduleID,"enable%sStream",moduleName);
  plugins[moduleID].disableStream = linkFunction(moduleID,"disable%sStream",moduleName);

  plugins[moduleID].passUserInput = linkFunction(moduleID,"passUserInput%s",moduleName);


  plugins[moduleID].getNumberOfDevices = linkFunction(moduleID,"get%sNumberOfDevices",moduleName);
  plugins[moduleID].getTotalFrameNumber =  linkFunction(moduleID,"getTotal%sFrameNumber",moduleName);
  plugins[moduleID].getCurrentFrameNumber = linkFunction(moduleID,"getCurrent%sFrameNumber",moduleName);
  plugins[moduleID].seekRelativeFrame = linkFunction(moduleID,"seekRelative%sFrame",moduleName);
  plugins[moduleID].seekFrame = linkFunction(moduleID,"seek%sFrame",moduleName);
  plugins[moduleID].controlFlow = linkFunction(moduleID,"control%sFlow",moduleName);


  plugins[moduleID].snapFrames = linkFunction(moduleID,"snap%sFrames",moduleName);

  plugins[moduleID].getLastColorTimestamp = linkFunction(moduleID,"getLast%sColorTimestamp",moduleName);
  plugins[moduleID].getLastDepthTimestamp = linkFunction(moduleID,"getLast%sDepthTimestamp",moduleName);

  plugins[moduleID].getNumberOfColorStreams = linkFunction(moduleID,"get%sNumberOfColorStreams",moduleName);
  plugins[moduleID].switchToColorStream = linkFunction(moduleID,"switch%sToColorStream",moduleName);

  plugins[moduleID].getColorWidth = linkFunction(moduleID,"get%sColorWidth",moduleName);
  plugins[moduleID].getColorHeight = linkFunction(moduleID,"get%sColorHeight",moduleName);
  plugins[moduleID].getColorDataSize = linkFunction(moduleID,"get%sColorDataSize",moduleName);
  plugins[moduleID].getColorChannels = linkFunction(moduleID,"get%sColorChannels",moduleName);
  plugins[moduleID].getColorBitsPerPixel = linkFunction(moduleID,"get%sColorBitsPerPixel",moduleName);
  plugins[moduleID].getColorPixels = linkFunction(moduleID,"get%sColorPixels",moduleName);

  //plugins[moduleID].getColorFocalLength = linkFunction(moduleID,"get%sColorFocalLength",moduleName);
  //plugins[moduleID].getColorPixelSize = linkFunction(moduleID,"get%sColorPixelSize",moduleName);

  plugins[moduleID].getColorCalibration = linkFunction(moduleID,"get%sColorCalibration",moduleName);

  plugins[moduleID].setColorCalibration = linkFunction(moduleID,"set%sColorCalibration",moduleName);


  plugins[moduleID].getDepthWidth = linkFunction(moduleID,"get%sDepthWidth",moduleName);
  plugins[moduleID].getDepthHeight = linkFunction(moduleID,"get%sDepthHeight",moduleName);
  plugins[moduleID].getDepthDataSize = linkFunction(moduleID,"get%sDepthDataSize",moduleName);
  plugins[moduleID].getDepthChannels = linkFunction(moduleID,"get%sDepthChannels",moduleName);
  plugins[moduleID].getDepthBitsPerPixel = linkFunction(moduleID,"get%sDepthBitsPerPixel",moduleName);
  plugins[moduleID].getDepthPixels = linkFunction(moduleID,"get%sDepthPixels",moduleName);

  //plugins[moduleID].getDepthFocalLength = linkFunction(moduleID,"get%sDepthFocalLength",moduleName);
  //plugins[moduleID].getDepthPixelSize = linkFunction(moduleID,"get%sDepthPixelSize",moduleName);
  //
  plugins[moduleID].getDepthCalibration = linkFunction(moduleID,"get%sDepthCalibration",moduleName);

  sprintf(functionNameStr,"set%sDepthCalibration",moduleName);
  plugins[moduleID].setDepthCalibration = linkFunction(moduleID,"set%sDepthCalibration",moduleName);


  return 1;
}

int unlinkPlugin(ModuleIdentifier moduleID)
{
    if (plugins[moduleID].handle==0) { return 1; }
    dlclose(plugins[moduleID].handle);
    plugins[moduleID].handle=0;
    return 1;
}
