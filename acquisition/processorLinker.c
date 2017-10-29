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


void * linkProcessorFunction(int moduleID,const char * functionName,const char * moduleName)
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



int linkToProcessor(const char * processorName,const char * processorLibPath ,  int processorID)
{
   processors[processorID].handle = dlopen (processorName, RTLD_LAZY);
   if (!processors[processorID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, processorName , processorLibPath , dlerror());
        return 0;
       }
    dlerror();    /* Clear any existing error */

  processors[processorID].initArgs     = linkProcessorFunction(processorID,"initArgs_%s",processorLibPath);
  processors[processorID].setConfigStr = linkProcessorFunction(processorID,"setConfigStr_%s",processorLibPath);
  processors[processorID].setConfigInt = linkProcessorFunction(processorID,"setConfigInt_%s",processorLibPath);
  processors[processorID].getDataOutput= linkProcessorFunction(processorID,"getDataOutput_%s",processorLibPath);
  processors[processorID].addDataInput = linkProcessorFunction(processorID,"addDataInput_%s",processorLibPath);
  processors[processorID].getDepth     = linkProcessorFunction(processorID,"getDepth_%s",processorLibPath);
  processors[processorID].getColor     = linkProcessorFunction(processorID,"getColor_%s",processorLibPath);

  processors[processorID].processData  = linkProcessorFunction(processorID,"processData_%s",processorLibPath);
  processors[processorID].cleanup      = linkProcessorFunction(processorID,"cleanup_%s",processorLibPath);
  processors[processorID].stop         = linkProcessorFunction(processorID,"stop_%s",processorLibPath);

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
   unsigned int processorID = 0;
      for (processorID=0; processorID<processorsLoaded; processorID++)
       {
          if (processors[processorID].stop!=0)
                   { (*processors[processorID].stop) (); }
          unlinkProcessor(processorID);
       }

  return 0;
}

int bringProcessorOnline(const char * processorName,const char * processorLibPath,unsigned int *loadedID,int argc,const char *argv[])
{
 fprintf(stderr,"bringProcessorOnline not implemented\n");
 unsigned int where2TryToLoad = processorsLoaded;
 if (  linkToProcessor(processorName,processorLibPath , where2TryToLoad  ) )
    {
     if ( (argc!=0) && (argv!=0) )
     { if (processors[where2TryToLoad].initArgs!=0)
             { (*processors[where2TryToLoad].initArgs) (argc,argv); }
     }

     ++processorsLoaded;
     return 1;
    }


 return 0;
}
