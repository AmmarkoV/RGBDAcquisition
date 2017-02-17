#include <stdio.h>
#include <stdlib.h>
#include "../../acquisition/Acquisition.h"
#include "../../tools/AmMatrix/matrixCalculations.h"


unsigned int devID=0;
ModuleIdentifier moduleID = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//

char inputname[512]={0};

void closeEverything()
{
 fprintf(stderr,"Gracefully closing everything .. ");
 //Stop our target ( can be network or files or nothing )
 acquisitionStopTargetForFrames(moduleID,devID);
 /*The first argument (Dev ID) could also be ANY_OPENNI2_DEVICE for a single camera setup */
 acquisitionCloseDevice(moduleID,devID);
 acquisitionStopModule(moduleID);

 fprintf(stderr,"Done\n");
 exit(0);
}


int main(int argc, char *argv[])
{
  fprintf(stderr,"TestBody started \n");
  acquisitionRegisterTerminationSignal(&closeEverything);

  unsigned int width=640,height=480,framerate=30;
  unsigned int frameNum=0;
  unsigned int maxFramesToGrab=0;

  unsigned int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-dev")==0)      {
                                           devID = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID);
                                         } else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname); }
             else
    if (strcmp(argv[i],"-fps")==0)       {
                                             framerate=atoi(argv[i+1]);
                                             fprintf(stderr,"Framerate , set to %u  \n",framerate);
                                         }

  }


  char * devName = inputname;
  if (strlen(inputname)<1) { devName=0; }
  if (!acquisitionOpenDevice(moduleID,devID,devName,width,height,framerate))
        {
          fprintf(stderr,"Could not open device %u ( %s ) of module %s  ..\n",devID,devName,getModuleNameFromModuleID(moduleID));
          return 1;
        }


   while  ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) )
    {
        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID,devID);

       // acquisitionPassFramesToTarget(moduleID,devID,frameNum,0);

        acquisitionStopTimer(0);
        if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
        ++frameNum;


        char outfilename[1024]={0};
         sprintf(outfilename,"%s/dnnOut/colorFrame_%u_%05u.json",inputname,devID,frameNum);
         fprintf(stderr," will read %s \n",outfilename);
         //acquisitionSaveColorFrame(moduleID,devID,outfilename);


    }

    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);

    closeEverything();

 return 0;
}
