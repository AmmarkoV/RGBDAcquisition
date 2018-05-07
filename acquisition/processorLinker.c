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
   processors[processorID].handle = dlopen (processorName, RTLD_NOW); //RTLD_LAZY
   if (!processors[processorID].handle)
       {
        fprintf (stderr,RED "Failed while loading code for %s plugin from %s\n Error : %s\n" NORMAL, processorName , processorLibPath , dlerror());


        char pathTester[4097]={0};
        if (getcwd(pathTester, 4096) != 0)
         fprintf(stdout, "  working dir was : %s\n", pathTester);


        return 0;
       }
    dlerror();    /* Clear any existing error */

  unsigned int missingCalls = 0;
  processors[processorID].initArgs     = linkProcessorFunction(processorID,"initArgs_%s",processorLibPath);
  missingCalls+=(processors[processorID].initArgs==0);

  processors[processorID].setConfigStr = linkProcessorFunction(processorID,"setConfigStr_%s",processorLibPath);
  missingCalls+=(processors[processorID].setConfigStr==0);

  processors[processorID].setConfigInt = linkProcessorFunction(processorID,"setConfigInt_%s",processorLibPath);
  missingCalls+=(processors[processorID].setConfigInt==0);

  processors[processorID].getDataOutput= linkProcessorFunction(processorID,"getDataOutput_%s",processorLibPath);
  missingCalls+=(processors[processorID].getDataOutput==0);

  processors[processorID].addDataInput = linkProcessorFunction(processorID,"addDataInput_%s",processorLibPath);
  missingCalls+=(processors[processorID].addDataInput==0);

  processors[processorID].getDepth     = linkProcessorFunction(processorID,"getDepth_%s",processorLibPath);
  missingCalls+=(processors[processorID].getDepth==0);

  processors[processorID].getColor     = linkProcessorFunction(processorID,"getColor_%s",processorLibPath);
  missingCalls+=(processors[processorID].getColor==0);

  processors[processorID].processData  = linkProcessorFunction(processorID,"processData_%s",processorLibPath);
  missingCalls+=(processors[processorID].processData==0);

  processors[processorID].cleanup      = linkProcessorFunction(processorID,"cleanup_%s",processorLibPath);
  missingCalls+=(processors[processorID].cleanup==0);

  processors[processorID].stop         = linkProcessorFunction(processorID,"stop_%s",processorLibPath);
  missingCalls+=(processors[processorID].stop==0);



  fprintf(stderr," Processor Loaded..\n");
  fprintf(stderr,"  Missing Calls = %u \n",missingCalls);
  /*
  fprintf(stderr," Processor Address Table\n");
  fprintf(stderr," initArgs=%p\n ",processors[processorID].initArgs  );
  fprintf(stderr," setConfigStr=%p\n ",processors[processorID].setConfigStr);
  fprintf(stderr," setConfigInt=%p\n ",processors[processorID].setConfigInt);
  fprintf(stderr," getDataOutput=%p\n ",processors[processorID].getDataOutput);
  fprintf(stderr," addDataInput=%p\n ",processors[processorID].addDataInput);
  fprintf(stderr," getDepth=%p\n ",processors[processorID].getDepth);
  fprintf(stderr," getColor=%p\n ",processors[processorID].getColor);

  fprintf(stderr," processData=%p\n ",processors[processorID].processData);
  fprintf(stderr," cleanup=%p\n ",processors[processorID].cleanup );
  fprintf(stderr," stop=%p\n ",processors[processorID].stop );
*/

  if (missingCalls==10)
  {
    fprintf(stderr,RED " All calls are missing, failed to load..\n " NORMAL);
    return 0;
  }


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
