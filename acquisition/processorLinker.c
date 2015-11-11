#include "processorLinker.h"

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


struct acquisitionProcessorInterface processors[MAX_NUMBER_OF_PROCESSORS];


void * linkProcessorFunction(int moduleID,char * functionName,char * moduleName)
{
  char functionNameStr[1024]={0};
  char *error;
  sprintf(functionNameStr,functionName,moduleName);
  void * linkPtr = dlsym(processors[moduleID].handle, functionNameStr );
  if ((error = dlerror()) != NULL)
     { fprintf (stderr, YELLOW "Could not find a definition of %s : %s\n" NORMAL ,functionNameStr ,  error); }
     //else { fprintf (stderr, GREEN "Found %s \n" NORMAL ,functionNameStr ); }

  return linkPtr;
}



int linkToProcessor(char * processorName,char * processorPossiblePath ,char * processorLib ,  int processorID)
{
    /*
   char *error;
   char functionNameStr[1024]={0};

   if (!getPluginPath(processorPossiblePath,processorLib,functionNameStr,1024))
       {
          fprintf(stderr,RED "Could not find %s (try adding it to current directory)\n" NORMAL , processorLib);
          return 0;
       }

   processors[processorID].handle = dlopen (functionNameStr, RTLD_LAZY);
   if (!processors[processorID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, processorName , functionNameStr , dlerror());
        return 0;
       }
*/
    dlerror();    /* Clear any existing error */


  processors[processorID].setConfigStr_DisparityMapping = linkProcessorFunction(processorID,"get%sDepthWidth",processorName);
  processors[processorID].setConfigInt_DisparityMapping = linkProcessorFunction(processorID,"get%sDepthHeight",processorName);
  processors[processorID].getDataOutput_DisparityMapping = linkProcessorFunction(processorID,"get%sDepthDataSize",processorName);
  processors[processorID].addDataInput_DisparityMapping = linkProcessorFunction(processorID,"get%sDepthChannels",processorName);
  processors[processorID].processData_DisparityMapping = linkProcessorFunction(processorID,"get%sDepthChannels",processorName);




  return 1;
}


int unlinkProcessor(int processorID)
{
  if (processors[processorID].handle==0) { return 1; }
  dlclose(processors[processorID].handle);
  processors[processorID].handle=0;
  return 1;
}
