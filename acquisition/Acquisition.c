#include "Acquisition.h"
#include "acquisition_setup.h"
#include "pluginLinker.h"
#include "processorLinker.h"
#include "acquisitionFileOutput.h"
#include "acquisitionScriptInput.h"

#include "../tools/Timers/timer.h"
#include "../tools/OperatingSystem/OperatingSystem.h"
#include "../tools/Primitives/modules.h"


   #if ENABLE_LOCATION_SERVICE
    #include "../tools/LocationServices/locationService.h"
   #endif // ENABLE_LOCATION_SERVICE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

char warnDryRun=0;


int useLocationServices=1; //By Default don't veto location services

unsigned int simulateTick=0;
unsigned long simulatedTickValue=0;


//This holds all the info on states of modules and devices
struct acquisitionModuleStates module[NUMBER_OF_POSSIBLE_MODULES]={0};



int acquisitionRegisterTerminationSignal(void * callback)
{
  return registerTerminationSignal(callback);
}

 int acquisitionCleanOverrides(ModuleIdentifier moduleID,DeviceIdentifier devID)
 {
   if (moduleID>=NUMBER_OF_POSSIBLE_MODULES) { return 0; }
   if (devID>=NUMBER_OF_POSSIBLE_DEVICES) { return 0; }

    if ( module[moduleID].device[devID].overrideColorFrame!=0 )
        {
          free(module[moduleID].device[devID].overrideColorFrame);
          module[moduleID].device[devID].overrideColorFrame=0;
        }

    if ( module[moduleID].device[devID].overrideDepthFrame!=0 )
        {
          free(module[moduleID].device[devID].overrideDepthFrame);
          module[moduleID].device[devID].overrideDepthFrame=0;
        }

    return 1;
 }

int acquisitionSimulateTime(unsigned long timeInMillisecs)
{
  simulateTick=1;
  simulatedTickValue=timeInMillisecs;
  return 1;
}

unsigned long acquisitionGetTickCount()
{
   if (simulateTick) { return simulatedTickValue; }
   return GetTickCountMilliseconds();
}

int acquisitionFileExists(char * filename)
{
  if (filename==0) { return 0; }
  FILE * fp  = fopen(filename,"r");
    if (fp!=0)
    {
      fclose(fp);
      return 1;
    }

  return 0;
}

int makepath(const char * path)
{
  if (path==0) { return 0; }
    char command[2048];
    sprintf(command,"mkdir -p %s",path);
    fprintf(stderr,"Executing .. %s \n",command);

    return system(command);
}


void countdownDelay(int seconds)
{
    if (seconds==0) { return; } //No delay do nothing!
    int secCounter=seconds;

    for (secCounter=seconds; secCounter>0; secCounter--)
    {
      fprintf(stderr,"%u\n",secCounter);
      sleepMilliseconds(1000); // Waiting a while for the glitch frames to pass

    }
    sleepMilliseconds(1000); // Waiting a while for the glitch frames to pass
}


void acquisitionStartTimer(unsigned int timerID)
{
    StartTimer(timerID);
}

unsigned int acquisitionStopTimer(unsigned int timerID)
{
    return EndTimer(timerID);
}

float acquisitionGetTimerFPS(unsigned int timerID)
{
    return GetFPSTimer(timerID);
}

// TODO : internal handler here..


int acquisitionIsModuleAvailiable(ModuleIdentifier moduleID)
{
  if (moduleID < NUMBER_OF_POSSIBLE_MODULES)
  {
   char tmp[1024];
   return getPluginPath (
                         getPluginStr(moduleID,PLUGIN_PATH_STR) ,
                         getPluginStr(moduleID,PLUGIN_LIBNAME_STR) ,
                         tmp,
                         1024
                         );
  }
  return 0;
}


int acquisitionGetModulesCount()
{
  unsigned int modules = 0;

  fprintf(stderr,"Querying for Acquisition plug-ins : \n");

  if ( acquisitionIsModuleAvailiable(V4L2_ACQUISITION_MODULE) )          { fprintf(stderr,GREEN "V4L2 module found \n" NORMAL);       ++modules; }
  if ( acquisitionIsModuleAvailiable(V4L2STEREO_ACQUISITION_MODULE) )    { fprintf(stderr,GREEN "V4L2Stereo module found \n" NORMAL); ++modules; }
  if ( acquisitionIsModuleAvailiable(OPENGL_ACQUISITION_MODULE) )        { fprintf(stderr,GREEN "OpenGL module found \n" NORMAL);     ++modules; }
  if ( acquisitionIsModuleAvailiable(TEMPLATE_ACQUISITION_MODULE) )      { fprintf(stderr,GREEN "Template module found \n" NORMAL);   ++modules; }
  if ( acquisitionIsModuleAvailiable(FREENECT_ACQUISITION_MODULE) )      { fprintf(stderr,GREEN "Freenect module found \n" NORMAL);   ++modules; }
  if ( acquisitionIsModuleAvailiable(OPENNI1_ACQUISITION_MODULE) )       { fprintf(stderr,GREEN "OpenNI1 module found \n" NORMAL);    ++modules; }
  if ( acquisitionIsModuleAvailiable(OPENNI2_ACQUISITION_MODULE) )       { fprintf(stderr,GREEN "OpenNI2 module found \n" NORMAL);    ++modules; }
  if ( acquisitionIsModuleAvailiable(NETWORK_ACQUISITION_MODULE) )       { fprintf(stderr,GREEN "Network module found \n" NORMAL);    ++modules; }
  if ( acquisitionIsModuleAvailiable(DEPTHSENSE_ACQUISITION_MODULE) )    { fprintf(stderr,GREEN "DepthSense module found \n" NORMAL); ++modules; }
  if ( acquisitionIsModuleAvailiable(DESKTOP_ACQUISITION_MODULE) )       { fprintf(stderr,GREEN "Desktop module found \n" NORMAL);    ++modules; }
  if ( acquisitionIsModuleAvailiable(REALSENSE_ACQUISITION_MODULE) )     { fprintf(stderr,GREEN "RealSense module found \n" NORMAL);  ++modules; }

  return modules;
}


ModuleIdentifier getModuleIdFromModuleName(const char * moduleName)
{
 ModuleIdentifier moduleID = 0;
   if (strcasecmp("FREENECT",moduleName)==0 )   { moduleID = FREENECT_ACQUISITION_MODULE; }   else
   if (strcasecmp("OPENNI",moduleName)==0 )     { moduleID = OPENNI1_ACQUISITION_MODULE;  }   else
   if (strcasecmp("OPENNI1",moduleName)==0 )    { moduleID = OPENNI1_ACQUISITION_MODULE;  }   else
   if (strcasecmp("OPENNI2",moduleName)==0 )    { moduleID = OPENNI2_ACQUISITION_MODULE;  }   else
   if (strcasecmp("OPENGL",moduleName)==0 )     { moduleID = OPENGL_ACQUISITION_MODULE;   }   else
   if (strcasecmp("V4L2",moduleName)==0 )       { moduleID = V4L2_ACQUISITION_MODULE;   }     else
   if (strcasecmp("V4L2STEREO",moduleName)==0 ) { moduleID = V4L2STEREO_ACQUISITION_MODULE; } else
   if (strcasecmp("TEMPLATE",moduleName)==0 )   { moduleID = TEMPLATE_ACQUISITION_MODULE; }   else
   if (strcasecmp("NETWORK",moduleName)==0 )    { moduleID = NETWORK_ACQUISITION_MODULE; }    else
   if (strcasecmp("DEPTHSENSE",moduleName)==0 ) { moduleID = DEPTHSENSE_ACQUISITION_MODULE; } else
   if (strcasecmp("DESKTOP",moduleName)==0 )    { moduleID = DESKTOP_ACQUISITION_MODULE; }    else
   if (strcasecmp("REALSENSE",moduleName)==0 )  { moduleID = REALSENSE_ACQUISITION_MODULE; }  else
   if (strcasecmp("SCRIPT",moduleName)==0 )     { moduleID = SCRIPTED_ACQUISITION_MODULE; }   else
   if (strcasecmp("SCRIPTED",moduleName)==0 )   { moduleID = SCRIPTED_ACQUISITION_MODULE; }

 return moduleID;
}


char * getModuleNameFromModuleID(ModuleIdentifier moduleID)
{
  switch (moduleID)
    {
      case V4L2_ACQUISITION_MODULE    :  return (char*) "V4L2 MODULE"; break;
      case V4L2STEREO_ACQUISITION_MODULE    :  return (char*) "V4L2STEREO MODULE"; break;
      case FREENECT_ACQUISITION_MODULE:  return (char*) "FREENECT MODULE"; break;
      case OPENNI1_ACQUISITION_MODULE :  return (char*) "OPENNI1 MODULE"; break;
      case OPENNI2_ACQUISITION_MODULE :  return (char*) "OPENNI2 MODULE"; break;
      case OPENGL_ACQUISITION_MODULE :  return (char*) "OPENGL MODULE"; break;
      case TEMPLATE_ACQUISITION_MODULE    :  return (char*) "TEMPLATE MODULE"; break;
      case NETWORK_ACQUISITION_MODULE    :  return (char*) "NETWORK MODULE"; break;
      case DEPTHSENSE_ACQUISITION_MODULE    :  return (char*) "DEPTHSENSE MODULE"; break;
      case DESKTOP_ACQUISITION_MODULE    :  return (char*) "DESKTOP MODULE"; break;
      case REALSENSE_ACQUISITION_MODULE    :  return (char*) "REALSENSE MODULE"; break;
      case SCRIPTED_ACQUISITION_MODULE    :  return (char*) "SCRIPTED MODULE"; break;
    };
    return (char*) "UNKNOWN MODULE";
}





void printCall(ModuleIdentifier moduleID,DeviceIdentifier devID,char * fromFunction , char * file , int line)
{
   #if PRINT_DEBUG_EACH_CALL
    fprintf(stderr,"called %s module %u , device %u  , file %s , line %d ..\n",fromFunction,moduleID,devID,file,line);
   #endif
}



void MeaningfullWarningMessage(ModuleIdentifier moduleFailed,DeviceIdentifier devFailed,char * fromFunction)
{
  if (!acquisitionIsModuleAvailiable(moduleFailed))
   {
       fprintf(stderr,"%s is not linked in this build of the acquisition library system..\n",getModuleNameFromModuleID(moduleFailed));
       return ;
   }

   fprintf(stderr,"%s hasn't got an implementation for function %s ..\n",getModuleNameFromModuleID(moduleFailed),fromFunction);
}



unsigned char * fastRGBDoubleSizer(unsigned char * rgb,unsigned int rgbWidth,unsigned int rgbHeight)
{
  #warning "fastRGBDoubleSizer can be improved "
  fprintf(stderr,"Will now double the incoming frame with dimensions %u x %u ( %u size ) \n" ,rgbWidth,rgbHeight , rgbWidth*rgbHeight*3);
  unsigned int doubleWidth = rgbWidth*2;
  unsigned int doubleHeight = rgbHeight*2;
  fprintf(stderr,"New image will be %u x %u ( %u size ) \n" ,doubleWidth,doubleHeight , doubleWidth*doubleHeight*3);

  unsigned char * result = (unsigned char*) malloc(sizeof(unsigned char)*3*doubleWidth*doubleHeight);


  if (result!=0)
  {
    unsigned char * pt2x        = result;
    unsigned char * pt2xLimit   = result + doubleWidth * doubleHeight *3;
    unsigned char * pt          = rgb;
    unsigned char * ptLineLimit = rgb + rgbWidth*3;
    unsigned char * ptLimit     = rgb + rgbWidth*rgbHeight*3;
    unsigned int line=0;
    unsigned char r,g,b;

    while  ( (pt2x<pt2xLimit)&&(pt<ptLimit) )
    {
     unsigned char * ptStartLine = pt;

     //First Line
     while(pt<ptLineLimit)
     {
      r=*pt; ++pt;
      g=*pt; ++pt;
      b=*pt; ++pt;

      *pt2x=r; ++pt2x;
      *pt2x=g; ++pt2x;
      *pt2x=b; ++pt2x;
      *pt2x=r; ++pt2x;
      *pt2x=g; ++pt2x;
      *pt2x=b; ++pt2x;
      }

     //Second Line
     pt = ptStartLine;
     while(pt<ptLineLimit)
     {
      r=*pt; ++pt;
      g=*pt; ++pt;
      b=*pt; ++pt;

      *pt2x=r; ++pt2x;
      *pt2x=g; ++pt2x;
      *pt2x=b; ++pt2x;
      *pt2x=r; ++pt2x;
      *pt2x=g; ++pt2x;
      *pt2x=b; ++pt2x;
      }

     ++line;
     ptLineLimit+=rgbWidth*3;
    }
  }
 return result;
}

/*! ------------------------------------------
    BASIC START STOP MECHANISMS FOR MODULES..
   ------------------------------------------*/


int acquisitionPluginIsLoaded(ModuleIdentifier moduleID)
{
 return isPluginLoaded(moduleID);
}

int acquisitionLoadPlugin(ModuleIdentifier moduleID)
{
  return   linkToPlugin(getPluginStr(moduleID,PLUGIN_NAME_STR),
                       getPluginStr(moduleID,PLUGIN_PATH_STR),
                       getPluginStr(moduleID,PLUGIN_LIBNAME_STR),
                       moduleID);
}

int acquisitionUnloadPlugin(ModuleIdentifier moduleID)
{
  return  unlinkPlugin(moduleID);
}

int acquisitionGetModuleCapabilities(ModuleIdentifier moduleID , DeviceIdentifier devID,int capToAskFor)
{
  if (*plugins[moduleID].getModuleCapabilities!=0) { return (*plugins[moduleID].getModuleCapabilities) (devID,capToAskFor); }

 MeaningfullWarningMessage(moduleID,0,"acquisitionGetModuleCapabilities");
 return 0;
}


int acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,const char * settings)
{
   #if ENABLE_LOCATION_SERVICE
   if (useLocationServices)
     { startLocationServices(); }
   #endif // ENABLE_LOCATION_SERVICE

  if (moduleID==SCRIPTED_ACQUISITION_MODULE)
  {
    //Special case for scripts
     fprintf(stderr,"Scripts don't need module start..\n");
    return 1;
  }


  if (moduleID < NUMBER_OF_POSSIBLE_MODULES)
  {
    if (!acquisitionLoadPlugin(moduleID))
        { fprintf(stderr,RED "Could not find %s plugin shared object \n" NORMAL,getModuleNameFromModuleID(moduleID)); return 0; }

    if (*plugins[moduleID].startModule!=0) { return (*plugins[moduleID].startModule) (maxDevices,settings); }
  } else
  {
    fprintf(stderr,RED "The plugin requested ( %s ) does not exist in the current plugin list , please recompile..\n" NORMAL,getModuleNameFromModuleID(moduleID));
  }

    MeaningfullWarningMessage(moduleID,0,"acquisitionStartModule");
    return 0;
}


int acquisitionStopModule(ModuleIdentifier moduleID)
{
   #if ENABLE_LOCATION_SERVICE
    if (useLocationServices)
     { stopLocationServices(); }
   #endif // ENABLE_LOCATION_SERVICE

   //Close all processors ..!
    closeAllProcessors();


    if (*plugins[moduleID].stopModule!=0) { return (*plugins[moduleID].stopModule) (); }
    acquisitionUnloadPlugin(moduleID);

    MeaningfullWarningMessage(moduleID,0,"acquisitionStopModule");
    return 0;
}


int acquisitionGetModuleDevices(ModuleIdentifier moduleID)
{
    printCall(moduleID,0,"acquisitionGetModuleDevices", __FILE__, __LINE__);
    if (plugins[moduleID].getNumberOfDevices!=0) { return (*plugins[moduleID].getNumberOfDevices) (); }
    MeaningfullWarningMessage(moduleID,0,"acquisitionGetModuleDevices");
    return 0;
}



int acquisitionMayBeVirtualDevice(ModuleIdentifier moduleID,DeviceIdentifier devID , char * devName)
{
  if ( (moduleID==OPENNI2_ACQUISITION_MODULE) && (acquisitionFileExists(devName)) )
      {
        fprintf(stderr,"Hoping that %s is a valid virtual device\n",devName);
       return 1;
      }

  return 0;
}




int acquisitionEnableStream(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int streamID)
{
    printCall(moduleID,0,"acquisitionEnableStream", __FILE__, __LINE__);
    if (plugins[moduleID].enableStream!=0) { return (*plugins[moduleID].enableStream) (devID,streamID); }
    MeaningfullWarningMessage(moduleID,0,"acquisitionEnableStream");
    return 0;
}

int acquisitionDisableStream(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int streamID)
{
    printCall(moduleID,0,"acquisitionDisableStream", __FILE__, __LINE__);
    if (plugins[moduleID].disableStream!=0) { return (*plugins[moduleID].disableStream) (devID,streamID); }
    MeaningfullWarningMessage(moduleID,0,"acquisitionDisableStream");
    return 0;
}



/*! ------------------------------------------
    FRAME SNAPPING MECHANISMS FOR MODULES..
   ------------------------------------------*/


int acquisitionListDevices(ModuleIdentifier moduleID,DeviceIdentifier devID,char * output, unsigned int maxOutput)
{
    printCall(moduleID,devID,"acquisitionListDevices", __FILE__, __LINE__);
    if (plugins[moduleID].listDevices!=0) { return (*plugins[moduleID].listDevices) (devID,output,maxOutput); }
    MeaningfullWarningMessage(moduleID,devID,"acquisitionListDevices");
    return 0;
}


int acquisitionChangeResolution(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int width,unsigned int height)
{
  printCall(moduleID,devID,"acquisitionChangeResolution", __FILE__, __LINE__);
  if (plugins[moduleID].changeResolution!=0) { return (*plugins[moduleID].changeResolution) (width,height); }

  plugins[moduleID].forcedWidth=width;
  plugins[moduleID].forcedHeight=height;
  plugins[moduleID].forceResolution=1;
  return 1;
}


int acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
    printCall(moduleID,devID,"acquisitionOpenDevice", __FILE__, __LINE__);

    if (moduleID==SCRIPTED_ACQUISITION_MODULE)
    {
     //Special case for scripts
     fprintf(stderr,"Script initialization starting..");
     return executeScriptFromFile(moduleID,devID,devName);
    }

    if (moduleID>=NUMBER_OF_POSSIBLE_MODULES) { MeaningfullWarningMessage(moduleID,devID,"Incorrect ModuleID"); return 0; }
    if (devID>=NUMBER_OF_POSSIBLE_DEVICES)    { MeaningfullWarningMessage(moduleID,devID,"Incorrect DeviceID"); return 0; }

    module[moduleID].device[devID].overrideColorFrame=0;
    module[moduleID].device[devID].overrideDepthFrame=0;

    if (plugins[moduleID].createDevice!=0) { return (*plugins[moduleID].createDevice) (devID,devName,width,height,framerate); }
    MeaningfullWarningMessage(moduleID,devID,"acquisitionOpenDevice");
    return 0;
}


 int acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    acquisitionCleanOverrides(moduleID,devID);
    printCall(moduleID,devID,"acquisitionCloseDevice", __FILE__, __LINE__);
    if (plugins[moduleID].destroyDevice!=0) { return (*plugins[moduleID].destroyDevice) (devID); }
    MeaningfullWarningMessage(moduleID,devID,"acquisitionCloseDevice");
    return 0;
}


int acquisitionGetTotalFrameNumber(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  printCall(moduleID,devID,"acquisitionGetTotalFrameNumber", __FILE__, __LINE__);
  if (*plugins[moduleID].getTotalFrameNumber!=0) { return (*plugins[moduleID].getTotalFrameNumber) (devID); }

  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetTotalFrameNumber");
  return 0;
}


int acquisitionPrepareDifferentResolutionFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    if ( plugins[moduleID].forceResolution )
    {
      fprintf(stderr,"TODO :");
      return 1;
    }

  return 0;
}

int acquisitionGetCurrentFrameNumber(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  printCall(moduleID,devID,"acquisitionGetCurrentFrameNumber", __FILE__, __LINE__);
  if (*plugins[moduleID].getCurrentFrameNumber!=0) { return (*plugins[moduleID].getCurrentFrameNumber) (devID); }

  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetCurrentFrameNumber");
  return 0;
}



 int acquisitionSeekFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int seekFrame)
{
    printCall(moduleID,devID,"acquisitionSeekFrame", __FILE__, __LINE__);
    if (*plugins[moduleID].seekFrame!=0) { return (*plugins[moduleID].seekFrame) (devID,seekFrame); }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSeekFrame");
    return 0;
}


 int acquisitionSeekRelativeFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,signed int seekFrame)
{
    printCall(moduleID,devID,"acquisitionSeekRelativeFrame", __FILE__, __LINE__);
    if (*plugins[moduleID].seekRelativeFrame!=0) { return (*plugins[moduleID].seekRelativeFrame) (devID,seekFrame); }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionSeekRelativeFrame");
    return 0;
}


int acquisitionControlFlow(ModuleIdentifier moduleID,DeviceIdentifier devID,float newFlowState)
{
    printCall(moduleID,devID,"acquisitionControlFlow", __FILE__, __LINE__);
    if (*plugins[moduleID].controlFlow!=0) { return (*plugins[moduleID].controlFlow) (devID,newFlowState); }

    MeaningfullWarningMessage(moduleID,devID,"acquisitionControlFlow");

  return 0;
}



int acquisitionDoProcessorSubsystem(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   //unsigned int processorsCalled=0;
   unsigned int colorWidth, colorHeight, colorChannels, colorBitsperpixel;
   unsigned int depthWidth, depthHeight, depthChannels, depthBitsperpixel;

   unsigned int processorID = 0;
      for (processorID=0; processorID<module[moduleID].device[devID].processorsLoaded; processorID++)
       {
          if (
               (processors[processorID].processData!=0) &&
               (processors[processorID].cleanup!=0) &&
               (processors[processorID].addDataInput!=0) &&
               (processors[processorID].getDepth!=0) &&
               (processors[processorID].getColor!=0)
             )
          {

           acquisitionGetColorFrameDimensions(moduleID,devID,&colorWidth,&colorHeight,&colorChannels,&colorBitsperpixel);
           acquisitionGetDepthFrameDimensions(moduleID,devID,&depthWidth,&depthHeight,&depthChannels,&depthBitsperpixel);

           (*processors[processorID].addDataInput) (0, acquisitionGetColorFrame(moduleID,devID) , colorWidth, colorHeight, colorChannels, colorBitsperpixel);
           (*processors[processorID].addDataInput) (1, acquisitionGetDepthFrame(moduleID,devID) , depthWidth, depthHeight, depthChannels, depthBitsperpixel);

           (*processors[processorID].processData) ();

            fprintf(stderr,"going to do further calls \n");

            unsigned short * depthFrame = (*processors[processorID].getDepth) ( &depthWidth, &depthHeight, &depthChannels, &depthBitsperpixel) ;
            unsigned char *  colorFrame = (*processors[processorID].getColor) ( &colorWidth, &colorHeight, &colorChannels, &colorBitsperpixel) ;


            fprintf(stderr,"colorFrame=%p , depthFrame=%p \n",colorFrame,depthFrame);

            if (colorFrame!=0)
                 { acquisitionOverrideColorFrame(moduleID , devID , colorFrame , colorWidth*colorHeight*colorChannels*(colorBitsperpixel/8) ,colorWidth,colorHeight,colorChannels,colorBitsperpixel); }

            if (depthFrame!=0)
                 { acquisitionOverrideDepthFrame(moduleID , devID , depthFrame , depthWidth*depthHeight*depthChannels*(depthBitsperpixel/8) ,depthWidth,depthHeight,depthChannels,depthBitsperpixel); }

            (*processors[processorID].cleanup) ();
          } else
          {
            fprintf(stderr,"Cannot run processor %u , it hasn't got all relevant calls implemented \n",processorID);
            fprintf(stderr," processData=%u ",(processors[processorID].processData!=0) );
            fprintf(stderr," cleanup=%u ",(processors[processorID].cleanup!=0) );
            fprintf(stderr," addDataInput=%u ",(processors[processorID].addDataInput!=0) );
            fprintf(stderr," getDepth=%u ",(processors[processorID].getDepth!=0) );
            fprintf(stderr," getColor=%u ",(processors[processorID].getColor!=0));
          }

       }



  return 1;
}




 int acquisitionSnapFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    printCall(moduleID,devID,"acquisitionSnapFrames", __FILE__, __LINE__);

    //In case the last frame had some overrides we "clear" them on our new frame so our new frame will not also appear overrided
    acquisitionCleanOverrides(moduleID,devID);

    StartTimer(FRAME_SNAP_DELAY);


   //TODO : only poll location services if we have a live stream
   #if ENABLE_LOCATION_SERVICE
    if (useLocationServices)
      { pollLocationServices(); }
   #endif // ENABLE_LOCATION_SERVICE

   EndTimer(FRAME_SNAP_DELAY);

   int snapResult = 0;
    if (*plugins[moduleID].snapFrames!=0)
    {
      snapResult = (*plugins[moduleID].snapFrames) (devID);
      acquisitionDoProcessorSubsystem(moduleID,devID);
    } else
    { MeaningfullWarningMessage(moduleID,devID,"acquisitionSnapFrames"); }

    return snapResult;
}




int savePCD_PointCloud(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy )
{
   return _acfo_savePCD_PointCloud(filename,depthFrame ,colorFrame,width,height,cx,cy,fx,fy);
}

int savePCD_PointCloudNoEmpty(char * filename ,unsigned short * depthFrame ,unsigned char * colorFrame , unsigned int width , unsigned int height , float cx , float cy , float fx , float fy )
{
  return _acfo_savePCD_PointCloudNoEmpty( filename ,depthFrame , colorFrame , width ,height ,cx ,cy ,fx ,fy );
}


int acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  return _acfo_acquisitionSavePCDPointCoud(moduleID,devID,filename);
}


int swapEndiannessPNM(void * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  return _acfo_swapEndiannessPNM( pixels , width , height , channels , bitsperpixel);

}


int acquisitionSaveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  return _acfo_acquisitionSaveRawImageToFile( filename, pixels , width , height , channels , bitsperpixel);
}


int acquisitionSaveLocationStamp(char * filename)
{
  return _acfo_acquisitionSaveLocationStamp( filename);
}


unsigned char * convertShortDepthToRGBDepth(unsigned short * depth,unsigned int width , unsigned int height)
{
  return _acfo_convertShortDepthToRGBDepth( depth,width , height);
}


unsigned char * convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  return _acfo_convertShortDepthToCharDepth( depth,width , height , min_depth , max_depth);
}


unsigned char * convertShortDepthTo3CharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  return _acfo_convertShortDepthTo3CharDepth(depth,width , height , min_depth , max_depth);
}


int acquisitionSaveTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  return _acfo_acquisitionSaveTimestamp(moduleID,devID,filename);
}


int acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename, int compress)
{
  return _acfo_acquisitionSaveColorFrame(moduleID,devID,filename, compress);
}


int acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename, int compress)
{
  return _acfo_acquisitionSaveDepthFrame(moduleID,devID,filename, compress);
}


int acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  return _acfo_acquisitionSaveColoredDepthFrame(moduleID,devID, filename);
}


int acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  return  _acfo_acquisitionSaveDepthFrame1C(moduleID,devID, filename);
}






int acquisitionGetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   printCall(moduleID,devID,"acquisitionGetColorCalibration", __FILE__, __LINE__);
   if (*plugins[moduleID].getColorCalibration!=0) { return (*plugins[moduleID].getColorCalibration) (devID,calib); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorCalibration");
   return 0;
}

int acquisitionGetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   printCall(moduleID,devID,"acquisitionGetDepthCalibration", __FILE__, __LINE__);
   if (*plugins[moduleID].getDepthCalibration!=0) { return (*plugins[moduleID].getDepthCalibration) (devID,calib); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthCalibration");
   return 0;
}





int acquisitionSetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   printCall(moduleID,devID,"acquisitionSetColorCalibration", __FILE__, __LINE__);
   if (*plugins[moduleID].setColorCalibration!=0) { return (*plugins[moduleID].setColorCalibration) (devID,calib); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorCalibration");
   return 0;
}

int acquisitionSetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib)
{
   printCall(moduleID,devID,"acquisitionSetDepthCalibration", __FILE__, __LINE__);
   if (*plugins[moduleID].setDepthCalibration!=0) { return (*plugins[moduleID].setDepthCalibration) (devID,calib); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthCalibration");
   return 0;
}


unsigned long acquisitionGetColorTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   printCall(moduleID,devID,"acquisitionGetColorTimestamp", __FILE__, __LINE__);
   if (*plugins[moduleID].getLastColorTimestamp!=0) { return (*plugins[moduleID].getLastColorTimestamp) (devID); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorTimestamp");
   return 0;
}

unsigned long acquisitionGetDepthTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
   printCall(moduleID,devID,"acquisitionGetDepthTimestamp", __FILE__, __LINE__);
   if (*plugins[moduleID].getLastDepthTimestamp!=0) { return (*plugins[moduleID].getLastDepthTimestamp) (devID); }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthTimestamp");
   return 0;
}


int acquisitionOverrideColorFrame(ModuleIdentifier moduleID , DeviceIdentifier devID , unsigned char * newColor , unsigned int newColorByteSize , unsigned int width ,unsigned int height , unsigned int channels, unsigned int bitsperpixel)
{
  if (newColor==0) { fprintf(stderr,"acquisitionOverrideColorFrame called with null new buffer");  return 0; }
  //Ok when someone overrides a color frame there are two chances..
  //He either has already done it ( there is already an overriden frame ) , Or it is the first time he does it ( there is NO overriden frame existing )
  //Since allocating/freeing chunks of memory means syscalls and since most of the time the size of each frame is the same in case there is an already
  //Overriden frame we just use the space allocated for it in order to be economic
  if ( module[moduleID].device[devID].overrideColorFrame!=0 )
  {
     fprintf(stderr,"Override Color Frame called while already overriden!\n");
     if ( module[moduleID].device[devID].overrideColorFrameByteSize < newColorByteSize )
     {
      fprintf(stderr,"Old override frame is not big enough to fit our frame , will not use it!\n");
      free(module[moduleID].device[devID].overrideColorFrame);
      module[moduleID].device[devID].overrideColorFrame=0;
      module[moduleID].device[devID].overrideColorFrameByteSize  = 0;
     } else
     {
      fprintf(stderr,"Old override frame is big enough , we will just use it instead of doing more syscalls to achieve the same result!\n");
     }
  }

  //In case the frame is not allocated ( or it was but it was not big enough ) , we have to malloc a new memory chunk to accomodate our frame
  if ( module[moduleID].device[devID].overrideColorFrame==0 )
  {
      module[moduleID].device[devID].overrideColorFrame = ( unsigned char * ) malloc(newColorByteSize*sizeof(unsigned char));
      if (module[moduleID].device[devID].overrideColorFrame==0)
      {
        module[moduleID].device[devID].overrideColorFrameByteSize  = 0;
        fprintf(stderr,"Could not allocate a new override frame of size %u\n",newColorByteSize);
        return 0;
      }
      module[moduleID].device[devID].overrideColorFrameByteSize  = newColorByteSize;
  }

  //Ok so after everything if we enter here it means we can <<at-last>> do our memcpy and get this over with
  if ( module[moduleID].device[devID].overrideColorFrame!=0 )
  {
      memcpy(module[moduleID].device[devID].overrideColorFrame , newColor , newColorByteSize);
      return 1;
  }

  return 0;
}


int acquisitionPassKeystroke(ModuleIdentifier moduleID , DeviceIdentifier devID, char key)
{
  //If getNumberOfColorStreams is declared for the plugin return its values
  if (*plugins[moduleID].passUserInput!=0)
       { return (*plugins[moduleID].passUserInput) (devID,key,0,0,0); } else

  //If we don't find it return 0
  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetNumberOfColorStreams");
  return 0;
}






int acquisitionGetNumberOfColorStreams(ModuleIdentifier moduleID , DeviceIdentifier devID)
{
  printCall(moduleID,devID,"acquisitionGetNumberOfColorStreams", __FILE__, __LINE__);
  //If getNumberOfColorStreams is declared for the plugin return its values
  if (*plugins[moduleID].getNumberOfColorStreams!=0) { return (*plugins[moduleID].getNumberOfColorStreams) (devID); } else
  //If we dont find a declared getNumberOfColorStreams BUT the getColorPixels is declared it looks like a missing function on the plugin
  //So we return 1 device ( the default
  if (*plugins[moduleID].getColorPixels!=0) { return 1; }

  //If we don't find it return 0
  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetNumberOfColorStreams");
  return 0;
}



int acquisitionSwitchToColorStream(ModuleIdentifier moduleID , DeviceIdentifier devID , unsigned int streamToActivate)
{
  printCall(moduleID,devID,"acquisitionSwitchToColorStream", __FILE__, __LINE__);
  if (*plugins[moduleID].switchToColorStream!=0) { return (*plugins[moduleID].switchToColorStream) (devID,streamToActivate); }
  MeaningfullWarningMessage(moduleID,devID,"acquisitionSwitchToColorStream");
  return 0;
}







unsigned char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  printCall(moduleID,devID,"acquisitionGetColorFrame", __FILE__, __LINE__);

  if (module[moduleID].device[devID].overrideColorFrame!=0) {
                                                              fprintf(stderr,"Returning overriden color frame \n");
                                                              return module[moduleID].device[devID].overrideColorFrame;
                                                            }

  if (*plugins[moduleID].getColorPixels!=0) { return (*plugins[moduleID].getColorPixels) (devID); }
  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorFrame");
  return 0;
}

unsigned int acquisitionCopyColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned char * mem,unsigned int memlength)
{
  printCall(moduleID,devID,"acquisitionCopyColorFrame", __FILE__, __LINE__);
  if ( (mem==0) || (memlength==0) )
  {
    fprintf(stderr,RED "acquisitionCopyColorFrame called with incorrect target for memcpy, %u bytes size" NORMAL,memlength);
    return 0;
  }


  unsigned char * color = acquisitionGetColorFrame(moduleID,devID);
  if (color==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
  unsigned int copySize = width*height*channels*(bitsperpixel/8);
  memcpy(mem,color,copySize);
  return copySize;
}


unsigned int acquisitionCopyColorFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned char * mem,unsigned int memlength)
{
  printCall(moduleID,devID,"acquisitionCopyColorFramePPM", __FILE__, __LINE__);
  if ( (mem==0) || (memlength==0) )
  {
    fprintf(stderr,RED "acquisitionCopyColorFramePPM called with incorrect target for memcpy, %u bytes size" NORMAL,memlength);
    return 0;
  }

  char * memC = (char*) mem;

  unsigned char * color = acquisitionGetColorFrame(moduleID,devID);
  if (color==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  sprintf(memC, "P6%d %d\n%u\n", width, height , _acfo_simplePow(2 ,bitsperpixel)-1);
  unsigned int payloadStart = strlen(memC);

  unsigned char * memPayload = mem + payloadStart ;
  memcpy(memPayload,color,width*height*channels*(bitsperpixel/8));

  payloadStart += width*height*channels*(bitsperpixel/8);
  return payloadStart;
}



int acquisitionOverrideDepthFrame(ModuleIdentifier moduleID , DeviceIdentifier devID , unsigned short * newDepth , unsigned int newDepthByteSize , unsigned int width ,unsigned int height , unsigned int channels, unsigned int bitsperpixel)
{
  if (newDepth==0) { fprintf(stderr,"acquisitionOverrideDepthFrame called with null new buffer");  return 0; }
  //Ok when someone overrides a color frame there are two chances..
  //He either has already done it ( there is already an overriden frame ) , Or it is the first time he does it ( there is NO overriden frame existing )
  //Since allocating/freeing chunks of memory means syscalls and since most of the time the size of each frame is the same in case there is an already
  //Overriden frame we just use the space allocated for it in order to be economic
  if ( module[moduleID].device[devID].overrideDepthFrame!=0 )
  {
     fprintf(stderr,"Override Color Frame called while already overriden!\n");
     if ( module[moduleID].device[devID].overrideDepthFrameByteSize < newDepthByteSize )
     {
      fprintf(stderr,"Old override frame is not big enough to fit our frame , will not use it!\n");
      free(module[moduleID].device[devID].overrideDepthFrame);
      module[moduleID].device[devID].overrideDepthFrame=0;
      module[moduleID].device[devID].overrideDepthFrameByteSize  = 0;
     } else
     {
      fprintf(stderr,"Old override frame is big enough , we will just use it instead of doing more syscalls to achieve the same result!\n");
     }
  }

  //In case the frame is not allocated ( or it was but it was not big enough ) , we have to malloc a new memory chunk to accomodate our frame
  if ( module[moduleID].device[devID].overrideDepthFrame==0 )
  {
      module[moduleID].device[devID].overrideDepthFrame = ( unsigned short * ) malloc(newDepthByteSize /* *sizeof(unsigned short) */ );
      if (module[moduleID].device[devID].overrideDepthFrame==0)
      {
        module[moduleID].device[devID].overrideDepthFrameByteSize  = 0;
        fprintf(stderr,"Could not allocate a new override frame of size %u\n",newDepthByteSize);
        return 0;
      }
      module[moduleID].device[devID].overrideDepthFrameByteSize  = newDepthByteSize;
  }

  //Ok so after everything if we enter here it means we can <<at-last>> do our memcpy and get this over with
  if ( module[moduleID].device[devID].overrideDepthFrame!=0 )
  {
      memcpy(module[moduleID].device[devID].overrideDepthFrame , newDepth , newDepthByteSize);
      return 1;
  }

  return 0;
}


unsigned short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  printCall(moduleID,devID,"acquisitionGetDepthFrame", __FILE__, __LINE__);

  if (module[moduleID].device[devID].overrideDepthFrame!=0) {
                                                              fprintf(stderr,"Returning overriden depth frame \n");
                                                              return module[moduleID].device[devID].overrideDepthFrame;
                                                            }

  if (*plugins[moduleID].getDepthPixels!=0) { return (unsigned short*) (*plugins[moduleID].getDepthPixels) (devID); }
  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthFrame");
  return 0;
}


unsigned int acquisitionCopyDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned short * mem,unsigned int memlength)
{
  printCall(moduleID,devID,"acquisitionCopyDepthFrame", __FILE__, __LINE__);
  if ( (mem==0) || (memlength==0) )
  {
    fprintf(stderr,RED "acquisitionCopyDepthFrame called with incorrect target for memcpy , %u bytes size" NORMAL,memlength);
    return 0;
  }

  unsigned short * depth = acquisitionGetDepthFrame(moduleID,devID);
  if (depth==0) { return 0; }
  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
  unsigned int copySize = width*height*channels*(bitsperpixel/8);
  memcpy(mem,depth,copySize);
  return copySize;
}


unsigned int acquisitionCopyDepthFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned short * mem,unsigned int memlength)
{
  printCall(moduleID,devID,"acquisitionCopyDepthFramePPM", __FILE__, __LINE__);
  if ( (mem==0) || (memlength==0) )
  {
    fprintf(stderr,RED "acquisitionCopyDepthFramePPM called with incorrect target for memcpy , %u bytes size" NORMAL,memlength);
    return 0;
  }

  unsigned short * depth = acquisitionGetDepthFrame(moduleID,devID);
  if (depth==0) { return 0; }

  unsigned int width , height , channels , bitsperpixel;
  acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);

  sprintf((char*) mem, "P5%d %d\n%u\n", width, height , _acfo_simplePow(2 ,bitsperpixel)-1);
  unsigned int payloadStart = strlen((char*) mem);

  unsigned short * memPayload = mem + payloadStart ;
  memcpy(memPayload,depth,width*height*channels*(bitsperpixel/8));

  payloadStart += width*height*channels*(bitsperpixel/8);
  return payloadStart;
}



int acquisitionGetColorRGBAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , unsigned char * R ,unsigned char * G , unsigned char * B )
{
    unsigned char * colorFrame = acquisitionGetColorFrame(moduleID,devID);
    if (colorFrame == 0 ) { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorRGBAtXY , getting color frame"); return 0; }

    unsigned int width; unsigned int height; unsigned int channels; unsigned int bitsperpixel;
    if (! acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel) )
        {  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorRGBAtXY getting depth frame dims"); return 0; }

    if ( (x2d>=width) || (y2d>=height) )
        { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorRGBAtXY incorrect 2d x,y coords"); return 0; }


    unsigned char * colorValue = colorFrame + ( (y2d * width *  channels)  + (x2d * channels ) );
    *R = *colorValue; ++colorValue;
    *G = *colorValue; ++colorValue;
    *B = *colorValue;

    return 1;
}



unsigned short acquisitionGetDepthValueAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d )
{
    unsigned short * depthFrame = acquisitionGetDepthFrame(moduleID,devID);
    if (depthFrame == 0 ) { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthValueAtXY , getting depth frame"); return 0; }

    unsigned int width; unsigned int height; unsigned int channels; unsigned int bitsperpixel;
    if (! acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel) )
        {  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthValueAtXY getting depth frame dims"); return 0; }

    if ( (x2d>=width) || (y2d>=height) )
        { MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthValueAtXY incorrect 2d x,y coords"); return 0; }


    unsigned short * depthValue = depthFrame + (y2d * width + x2d );
    unsigned short result = * depthValue;

    return result;
}



int acquisitionGetDepth3DPointAtXYCameraSpaceWithCalib(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  )
{
    unsigned short depthValue = acquisitionGetDepthValueAtXY(moduleID,devID,x2d,y2d);
    if (depthValue==0) { fprintf(stderr,"acquisitionGetDepth3DPointAtXYCameraSpace point %u,%u has no depth\n",x2d,y2d); return 0; }

    return transform2DProjectedPointTo3DPoint(calib , x2d , y2d  , depthValue , x , y , z);
}

int acquisitionGetDepth3DPointAtXYCameraSpace(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  )
{
    struct calibration calib;
    unsigned short depthValue = acquisitionGetDepthValueAtXY(moduleID,devID,x2d,y2d);
    if (depthValue==0) { fprintf(stderr,"acquisitionGetDepth3DPointAtXYCameraSpace point %u,%u has no depth\n",x2d,y2d); return 0; }
    if ( !acquisitionGetDepthCalibration(moduleID,devID,&calib) )
        {
          if ( !acquisitionGetColorCalibration(moduleID,devID,&calib) )
          { fprintf(stderr,"Could not get Depth Calibration , cannot get 3D point\n"); return 0; }
        }

    return transform2DProjectedPointTo3DPoint(&calib , x2d , y2d  , depthValue , x , y , z);
}


int acquisitionGetDepth3DPointAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  )
{
    struct calibration calib;
    unsigned short depthValue = acquisitionGetDepthValueAtXY(moduleID,devID,x2d,y2d);
    if (depthValue==0) { fprintf(stderr,"acquisitionGetDepth3DPointAtXY point %u,%u has no depth\n",x2d,y2d); return 0; }
    if ( !acquisitionGetDepthCalibration(moduleID,devID,&calib) )
        {
          if ( !acquisitionGetColorCalibration(moduleID,devID,&calib) )
          { fprintf(stderr,"Could not get Depth Calibration , cannot get 3D point\n"); return 0; }
        }

    transform2DProjectedPointTo3DPoint(&calib , x2d , y2d  , depthValue , x , y , z);

    if (calib.extrinsicParametersSet)
    {
      return transform3DPointUsingCalibration(&calib , x , y , z);
    }
    return 1;
}

int acquisitionGetColorFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID ,
                                       unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel )
{
  printCall(moduleID,devID,"acquisitionGetColorFrameDimensions", __FILE__, __LINE__);

  if ( (width==0)||(height==0)||(channels==0)||(bitsperpixel==0) )
    {
        fprintf(stderr,"acquisitionGetColorFrameDimensions called with invalid arguments .. \n");
        return 0;
    }


         if (
              (*plugins[moduleID].getColorWidth!=0) && (*plugins[moduleID].getColorHeight!=0) &&
              (*plugins[moduleID].getColorChannels!=0) && (*plugins[moduleID].getColorBitsPerPixel!=0)
            )
            {
              *width        = (*plugins[moduleID].getColorWidth)        (devID);
              *height       = (*plugins[moduleID].getColorHeight)       (devID);
              *channels     = (*plugins[moduleID].getColorChannels)     (devID);
              *bitsperpixel = (*plugins[moduleID].getColorBitsPerPixel) (devID);
              return 1;
            }

  MeaningfullWarningMessage(moduleID,devID,"acquisitionGetColorFrameDimensions");
  return 0;
}



int acquisitionGetDepthFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID ,
                                       unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel )
{
  printCall(moduleID,devID,"acquisitionGetDepthFrameDimensions", __FILE__, __LINE__);

  if ( (width==0)||(height==0)||(channels==0)||(bitsperpixel==0) )
    {
        fprintf(stderr,"acquisitionGetDepthFrameDimensions called with invalid arguments .. \n");
        return 0;
    }


         if (
              (*plugins[moduleID].getDepthWidth!=0) && (*plugins[moduleID].getDepthHeight!=0) &&
              (*plugins[moduleID].getDepthChannels!=0) && (*plugins[moduleID].getDepthBitsPerPixel!=0)
            )
            {
              *width        = (*plugins[moduleID].getDepthWidth)        (devID);
              *height       = (*plugins[moduleID].getDepthHeight)       (devID);
              *channels     = (*plugins[moduleID].getDepthChannels)     (devID);
              *bitsperpixel = (*plugins[moduleID].getDepthBitsPerPixel) (devID);
              return 1;
            }
   MeaningfullWarningMessage(moduleID,devID,"acquisitionGetDepthFrameDimensions");
   return 0;
}


 int acquisitionMapDepthToRGB(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    printCall(moduleID,devID,"acquisitionMapDepthToRGB", __FILE__, __LINE__);
    if  (*plugins[moduleID].mapDepthToRGB!=0) { return  (*plugins[moduleID].mapDepthToRGB) (devID); }
    MeaningfullWarningMessage(moduleID,devID,"acquisitionMapDepthToRGB");
    return 0;
}


 int acquisitionMapRGBToDepth(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
    printCall(moduleID,devID,"acquisitionMapRGBToDepth", __FILE__, __LINE__);
    if  (*plugins[moduleID].mapRGBToDepth!=0) { return  (*plugins[moduleID].mapRGBToDepth) (devID); }
    MeaningfullWarningMessage(moduleID,devID,"acquisitionMapRGBToDepth");
    return 0;
}


/*
   LAST BUT NOT LEAST acquisition can also relay its state through a TCP/IP network
*/
int acquisitionInitiateTargetForFrames(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * target)
{
  if (strstr(target,"/dev/null")!=0)
  {
    module[moduleID].device[devID].dryRunOutput=1;
    return 1;
  } else
  if (strstr(target,"shm://")!=0)
  {
    module[moduleID].device[devID].sharedMemoryOutput=1;
    return 1;
  } else
  if ( strstr(target,"tcp://")!=0 )
  {
    if (
         !linkToNetworkTransmission(
                                     getPluginStr(NETWORK_ACQUISITION_MODULE,PLUGIN_NAME_STR) ,
                                     getPluginStr(NETWORK_ACQUISITION_MODULE,PLUGIN_PATH_STR)  ,
                                     getPluginStr(NETWORK_ACQUISITION_MODULE,PLUGIN_LIBNAME_STR)
                                   )
        )
      {
        fprintf(stderr,RED "Cannot link to network transmission framework , so will not be able to transmit output..!\n" NORMAL);
      } else
      {
       if  (*startPushingToRemoteNetwork!=0)
         {
            unsigned int width ,height , channels , bitsperpixel;
            acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
            module[moduleID].device[devID].frameServerID = (*startPushingToRemoteNetwork) ("0.0.0.0",1234,width ,height);
            module[moduleID].device[devID].networkOutput=1;
            return 1;
         }
      }
  } else
  {
    module[moduleID].device[devID].fileOutput=1;
    strcpy(module[moduleID].device[devID].outputString , target);

    //fprintf(stderr,"acquisitionInitiateTargetForFrames! Module %u , Device %u = %s \n",moduleID,devID, module[moduleID].device[devID].outputString);
    //Prepare path
    makepath(target);
    //Store Calibration here..

    char finalTarget[2048];
    struct calibration calibRGB={0};
    struct calibration calibDepth={0};

    acquisitionGetColorCalibration(moduleID,devID,&calibRGB);
    snprintf(finalTarget,2048,"%s/color.calib",target);
    WriteCalibration(finalTarget,&calibRGB);

    acquisitionGetDepthCalibration(moduleID,devID,&calibDepth);
    snprintf(finalTarget,2048,"%s/depth.calib",target);
    WriteCalibration(finalTarget,&calibDepth);


    return 1;
  }

  fprintf(stderr,RED "acquisitionInitiateTargetForFrames did not decide on a method for passing frames to target\n" NORMAL);
  return 0;
}


int acquisitionStopTargetForFrames(ModuleIdentifier moduleID,DeviceIdentifier devID)
{
  if (module[moduleID].device[devID].networkOutput)
  {
    //If we were using networkOutput lets stop using it!
    if (stopPushingToRemoteNetwork==0) { fprintf(stderr,RED "stopPushingToRemoteNetwork has not been linked to network plugin\n" NORMAL); return 0; }
    return (*stopPushingToRemoteNetwork) (module[moduleID].device[devID].frameServerID);
  } else
  if (module[moduleID].device[devID].sharedMemoryOutput)
  {
    fprintf(stderr,RED "acquisitionStopTargetForFrames implementation for shared memory pls\n" NORMAL);
  }
  return 1;
}



int acquisitionPassFramesToTarget(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int frameNumber,int doCompression)
{
  //fprintf(stderr,"acquisitionPassFramesToTarget not fully implemented yet! Module %u , Device %u = %s \n",moduleID,devID, module[moduleID].device[devID].outputString);

  if (module[moduleID].device[devID].dryRunOutput)
  {
    if (!warnDryRun)
     { fprintf(stderr,GREEN "Grabber Running on Dry Run mode , grabbing frames and doing nothing [ This message appears only one time ]\n" NORMAL); warnDryRun=1; }

    //Lets access the frames like we want to use them but not use them..

    StartTimer(FRAME_PASS_TO_TARGET_DELAY);
    unsigned int width, height , channels , bitsperpixel;
    acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    acquisitionGetColorFrame(moduleID,devID);

    acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    acquisitionGetDepthFrame(moduleID,devID);
    //Done doing nothing with our input..!
    EndTimer(FRAME_PASS_TO_TARGET_DELAY);
  } else
  if (module[moduleID].device[devID].sharedMemoryOutput)
  {
     fprintf(stderr,RED "Implement shm transmission here..\n" NORMAL);
  } else
  if (module[moduleID].device[devID].fileOutput)
  {
   StartTimer(FRAME_PASS_TO_TARGET_DELAY);
   char outfilename[2048]={0};

    sprintf(outfilename,"%s/colorFrame_%u_%05u",module[moduleID].device[devID].outputString,devID,frameNumber);
    acquisitionSaveColorFrame(moduleID,devID,outfilename,doCompression);

    sprintf(outfilename,"%s/depthFrame_%u_%05u",module[moduleID].device[devID].outputString,devID,frameNumber);
    acquisitionSaveDepthFrame(moduleID,devID,outfilename,doCompression);

    sprintf(outfilename,"%s/info_%u_%05u",module[moduleID].device[devID].outputString,devID,frameNumber);
    acquisitionSaveTimestamp(moduleID,devID,outfilename);

   EndTimer(FRAME_PASS_TO_TARGET_DELAY);

  } else
  if (module[moduleID].device[devID].networkOutput)
  {
    StartTimer(FRAME_PASS_TO_TARGET_DELAY);

    unsigned int width, height , channels , bitsperpixel;
    acquisitionGetColorFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    pushImageToRemoteNetwork(module[moduleID].device[devID].frameServerID , 0 , (void*) acquisitionGetColorFrame(moduleID,devID) , width , height , channels , bitsperpixel);

    acquisitionGetDepthFrameDimensions(moduleID,devID,&width,&height,&channels,&bitsperpixel);
    pushImageToRemoteNetwork(module[moduleID].device[devID].frameServerID , 1 , (void*) acquisitionGetDepthFrame(moduleID,devID) , width , height , channels , bitsperpixel);

    EndTimer(FRAME_PASS_TO_TARGET_DELAY);
  } else
  {
    fprintf(stderr,RED "acquisitionPassFramesToTarget cannot find a method to use for module %u , device %u , has acquisitionInitiateTargetForFrames been called?\n" NORMAL , moduleID , devID );
    return 0;
  }

  return 1;
}




int acquisitionAddProcessor(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * processorName,const char * processorLibPath,int argc,const char *argv[])
{
  unsigned int weSucceeded=0;
  unsigned int where2LoadProcessor = module[moduleID].device[devID].processorsLoaded;
  if (bringProcessorOnline(processorName,processorLibPath, &where2LoadProcessor,argc,argv))
  {
   ++module[moduleID].device[devID].processorsLoaded;
   module[moduleID].device[devID].processorsIDs[where2LoadProcessor] = where2LoadProcessor;
   weSucceeded=1;
  } else
  {
   fprintf(stderr,RED "acquisitionAddProcessor could not load %s \n" NORMAL , processorName );
  }

 return weSucceeded;
}




int acquisitionSetLocation(ModuleIdentifier moduleID,int newState)
{
 fprintf(stderr,"acquisitionSetLocation(%u)\n",newState);
 useLocationServices=newState;
 plugins[moduleID].useLocationServicesForThisModule=newState;
 return 1;
}
