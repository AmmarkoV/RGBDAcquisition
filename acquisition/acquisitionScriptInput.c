#include "acquisitionScriptInput.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/TrajectoryParser/InputParser_C.h"

int acquisitionExecuteString(struct InputParserC * ipc,struct acquisitionModuleStates * state, ModuleIdentifier moduleID,DeviceIdentifier devID,const char * command)
{
  if (command==0) { return 0; }
  InputParser_SeperateWordsCC(ipc,command,1);

  char tag[512]={0};
  char moduleString[512]={0};
  InputParser_GetWord(ipc,0,tag,512);

  if (strcmp(tag,"acquisitionStartModule")==0)
  {
    fprintf(stderr,"acquisitionStartModule\n");
    InputParser_GetUpcaseWord(ipc,1,moduleString,512);
    unsigned int selectedModuleID=getModuleIdFromModuleName(moduleString);
    InputParser_GetWord(ipc,1,state[moduleID].device[devID].configuration,102);

    state[moduleID].device[devID].redirectModuleID = selectedModuleID;
    state[moduleID].device[devID].redirectDeviceID = 0;
    state[moduleID].device[devID].useRedirect=1; //Use redirects

    //acquisitionStartModule(selectedModuleID,1,state[moduleID].device[devID].configuration);
  } else
  if (strcmp(tag,"acquisitionOpenDevice")==0)
  {
    fprintf(stderr,"acquisitionOpenDevice\n");
    InputParser_GetUpcaseWord(ipc,1,moduleString,512);
    unsigned int selectedModuleID=getModuleIdFromModuleName(moduleString);



    unsigned int selectedDeviceID=InputParser_GetWordInt(ipc,2);
    InputParser_GetWord(ipc,3,state[moduleID].device[devID].deviceName,1024);

    state[moduleID].device[devID].redirectModuleID = selectedModuleID;
    state[moduleID].device[devID].redirectDeviceID = selectedDeviceID;
    state[moduleID].device[devID].useRedirect=1; //Use redirects


    state[moduleID].device[devID].width=InputParser_GetWordInt(ipc,4);
    state[moduleID].device[devID].height=InputParser_GetWordInt(ipc,5);
    state[moduleID].device[devID].framerate=InputParser_GetWordInt(ipc,6);

    /*
    acquisitionOpenDevice(
                          selectedModuleID,
                          selectedDeviceID,
                          state[moduleID].device[devID].deviceName,
                          state[moduleID].device[devID].width,
                          state[moduleID].device[devID].height,
                          state[moduleID].device[devID].framerate
                         );*/
  } else
  {
    if (command[0]!='#')
         { fprintf(stderr,"unknown line : %s \n",command); }
  }
return 1;
}


int getRealModuleAndDevice(
                           struct acquisitionModuleStates * state,
                           ModuleIdentifier * moduleID ,
                           DeviceIdentifier * devID ,
                           unsigned int * width ,
                           unsigned int * height,
                           unsigned int * framerate,
                           char * configuration,
                           char * deviceName,
                           unsigned int stringMaxLength
                          )
{
    unsigned int scModuleID=*moduleID;
    unsigned int scDevID=*devID;

    if (state[scModuleID].device[scDevID].useRedirect)
    {
     *moduleID  = state[scModuleID].device[scDevID].redirectModuleID;
     *devID     = state[scModuleID].device[scDevID].redirectDeviceID;

     *width     = state[scModuleID].device[scDevID].width;
     *height    = state[scModuleID].device[scDevID].height;
     *framerate = state[scModuleID].device[scDevID].framerate;

     if (configuration!=0)
     {
       snprintf(configuration,stringMaxLength,"%s",state[scModuleID].device[scDevID].configuration);
     }

     if (deviceName!=0)
     {
       snprintf(deviceName,stringMaxLength,"%s",state[scModuleID].device[scDevID].deviceName);
     }

     return (*moduleID!=SCRIPTED_ACQUISITION_MODULE);
    }
  return 0;
}



int executeScriptFromFile(struct acquisitionModuleStates * state,ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
{
  ssize_t read;

  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
    struct InputParserC * ipc = InputParser_Create(2048,4);

    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
     acquisitionExecuteString(ipc,state,moduleID,devID,line);
    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }
    return 1;
  }

 fprintf(stderr,"Could not execute acquisition script from %s\n",filename);

 char workingPath[4096]={0};
   if (getcwd(workingPath, 4096) != 0)
         fprintf(stdout, "  Working directory was : %s\n", workingPath);
 return 0;
}
