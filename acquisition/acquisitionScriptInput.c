#include "acquisitionScriptInput.h"
#include <stdio.h>
#include <stdlib.h>

#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.h"

int acquisitionExecuteString(struct InputParserC * ipc,ModuleIdentifier moduleID,DeviceIdentifier devID,const char * command)
{
  InputParser_SeperateWordsCC(ipc,command,1);

  char tag[512];
  char moduleString[512];
  char configString[512];
  char deviceString[512];
  InputParser_GetLowercaseWord(ipc,0,tag,512);
  if (strcmp(tag,"acquisitionStartModule")==0)
  {
    fprintf(stderr,"acquisitionStartModule\n");
    InputParser_GetLowercaseWord(ipc,1,moduleString,512);
    unsigned int selectedModuleID=getModuleIdFromModuleName(moduleString);
    acquisitionStartModule(selectedModuleID,1,configString);
  } else
  if (strcmp(tag,"acquisitionOpenDevice")==0)
  {
    fprintf(stderr,"acquisitionOpenDevice\n");
    InputParser_GetLowercaseWord(ipc,1,moduleString,512);
    unsigned int selectedModuleID=getModuleIdFromModuleName(moduleString);
    unsigned int selectedDeviceID=InputParser_GetWordInt(ipc,2);
    InputParser_GetLowercaseWord(ipc,3,deviceString,512);
    unsigned int width,height,framerate;


    width=InputParser_GetWordInt(ipc,4);
    height=InputParser_GetWordInt(ipc,5);
    framerate=InputParser_GetWordInt(ipc,6);

    acquisitionOpenDevice(selectedModuleID,selectedDeviceID,deviceString,width,height,framerate);
  }
return 1;
}



int executeScriptFromFile(ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename)
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
     acquisitionExecuteString(ipc,moduleID,devID,line);
    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }
    return 1;
  }
 return 0;
}
