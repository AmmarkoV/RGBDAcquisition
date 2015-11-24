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

unsigned int processorsLoaded=0;
struct acquisitionProcessorInterface processors[MAX_NUMBER_OF_PROCESSORS]={0};


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



int linkToProcessor(char * processorName,char * processorLibPath ,  int processorID)
{
   processors[processorID].handle = dlopen (processorName, RTLD_LAZY);
   if (!processors[processorID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, processorName , processorLibPath , dlerror());
        return 0;
       }
    dlerror();    /* Clear any existing error */

  processors[processorID].setConfigStr = linkProcessorFunction(processorID,"setConfigStr_%s",processorLibPath);
  processors[processorID].setConfigInt = linkProcessorFunction(processorID,"setConfigInt_%s",processorLibPath);
  processors[processorID].getDataOutput= linkProcessorFunction(processorID,"getDataOutput_%s",processorLibPath);
  processors[processorID].addDataInput = linkProcessorFunction(processorID,"addDataInput_%s",processorLibPath);
  processors[processorID].getDepth     = linkProcessorFunction(processorID,"getDepth_%s",processorLibPath);
  processors[processorID].getColor     = linkProcessorFunction(processorID,"getColor_%s",processorLibPath);

  processors[processorID].processData  = linkProcessorFunction(processorID,"processData_%s",processorLibPath);
  processors[processorID].cleanup  = linkProcessorFunction(processorID,"cleanup_%s",processorLibPath);

  return 1;
}


int unlinkProcessor(int processorID)
{
  if (processors[processorID].handle==0) { return 1; }
  dlclose(processors[processorID].handle);
  processors[processorID].handle=0;
  return 1;
}

int closeAllProcessors()
{
 fprintf(stderr,"closeAllProcessors not implemented\n");
  return 0;
}

int bringProcessorOnline(char * processorName,char * processorLibPath,unsigned int *loadedID)
{
 fprintf(stderr,"bringProcessorOnline not implemented\n");
 unsigned int where2TryToLoad = processorsLoaded;
 if (  linkToProcessor(processorName,processorLibPath , where2TryToLoad  ) )
    {
     ++processorsLoaded;
     return 1;
    }


 return 0;
}
