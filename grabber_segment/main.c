/** @file main.c
 *  @brief The simple Grabber that uses libAcquisition.so and libAcquisitionSegment.so and writes output to files
 *
 *  This should be used like ./GrabberSegment -module TEMPLATE -from Dataset -maxFrames 100 -to OutputDataset -eraseRGB 255 255 255 -maxDepth 1450 -minDepth 700 -plane -446.87997 180.98639 1074.00000 605.03101 79.81260 1236.00000 -335.44980 305.72638 684.00000 -floodEraseRGBSource 383 408 30 -cropRGB 164 0 481 395 -combine and
 *  For a full list of the command line parameters see the loadSegmentationDataFromArgs of AcquisitionSegment.c that populates the segmentation structures
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug No known bugs
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../acquisitionSegment/AcquisitionSegment.h"

unsigned int seekFrame=0;
int stereoSplit=0;
int calibrationSet = 0;
char calibrationFile[2048]={0};
struct calibration calib;

char outputfoldername[512]={0};
char inputname[512]={0};


//We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
unsigned int frameNum=0,maxFramesToGrab=10;
unsigned int devID_1=0 ;
ModuleIdentifier moduleID_1 = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//


void closeEverything()
{
 fprintf(stderr,"Gracefully closing everything .. ");

    acquisitionCloseDevice(moduleID_1,devID_1);

    acquisitionStopModule(moduleID_1);

 fprintf(stderr,"Done\n");
 exit(0);
}

int makepath(char * path)
{
    //FILE *fp;
    /* Open the command for reading. */
    char command[1024];
    sprintf(command,"mkdir -p %s",outputfoldername);
    fprintf(stderr,"Executing .. %s \n",command);

    return system(command);
}

unsigned char * copyRGB(unsigned char * source , unsigned int width , unsigned int height)
{
  if ( (source==0)  || (width==0) || (height==0) )
    {
      fprintf(stderr,"copyRGB called with zero arguments\n");
      return 0;
    }

  unsigned char * output = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (output==0) { fprintf(stderr,"copyRGB could not allocate memory for output\n"); return 0; }
  memcpy(output , source , width*height*3*sizeof(unsigned char));
  return output;
}

unsigned short * copyDepth(unsigned short * source , unsigned int width , unsigned int height)
{
  if ( (source==0)  || (width==0) || (height==0) )
    {
      fprintf(stderr,"copyDepth called with zero arguments\n");
      return 0;
    }

  unsigned short * output = (unsigned short*) malloc(width*height*sizeof(unsigned short));
  if (output==0) { fprintf(stderr,"copyDepth could not allocate memory for output\n"); return 0; }
  memcpy(output , source , width*height*sizeof(unsigned short));
  return output;
}




int doStereoSplit(
                  unsigned int frameNum,
                  unsigned char * rgb,
                  unsigned int widthRGB,
                  unsigned int heightRGB,
                  unsigned int channelsRGB,
                  unsigned int bitsperpixelRGB,
                  unsigned long colorTimestamp,
                  char * outputfoldername
                  )
{
  if (rgb==0) { return 0; }
  acquisitionSimulateTime( colorTimestamp );

  char outfilename[512]={0};

  unsigned int newWidth=widthRGB;
  unsigned int newHeight=heightRGB;
  unsigned char* ss = splitStereo(rgb,
                                  0,
                                  &newWidth,
                                  &newHeight
                                 );
  if (ss!=0)
  {
   sprintf(outfilename,"%s/colorFrame_0_%05u.pnm",outputfoldername,frameNum);
   acquisitionSaveRawImageToFile(outfilename,ss,newWidth,newHeight,channelsRGB,bitsperpixelRGB);
   sprintf(outfilename,"%s/left%02u.ppm",outputfoldername,frameNum);
   acquisitionSaveRawImageToFile(outfilename,ss,newWidth,newHeight,channelsRGB,bitsperpixelRGB);
   free(ss);
  }

  newWidth=widthRGB;
  newHeight=heightRGB;
  ss = splitStereo(rgb,
                   0,
                   &newWidth,
                   &newHeight
                   );
  if (ss!=0)
  {
   sprintf(outfilename,"%s/colorFrame_1_%05u.pnm",outputfoldername,frameNum);
   acquisitionSaveRawImageToFile(outfilename,ss,newWidth,newHeight,channelsRGB,bitsperpixelRGB);
   sprintf(outfilename,"%s/right%02u.ppm",outputfoldername,frameNum);
   acquisitionSaveRawImageToFile(outfilename,ss,newWidth,newHeight,channelsRGB,bitsperpixelRGB);
   free(ss);
  }

 return 1;
}




int main(int argc, char *argv[])
{
 fprintf(stderr,"Generic Multiplexed Grabber Application based on Acquisition lib .. \n");
 unsigned int possibleModules = acquisitionGetModulesCount();
 fprintf(stderr,"Linked to %u modules.. \n",possibleModules);

 acquisitionRegisterTerminationSignal(&closeEverything);


 if (possibleModules==0)
    {
       fprintf(stderr,"Acquisition Library is linked to zero modules , can't possibly do anything..\n");
       return 1;
    }


  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS --------------------------------- */


   //------------------------------------------------------------------
   //                        CONFIGURATION
   //------------------------------------------------------------------

   unsigned int combinationMode=DONT_COMBINE;

   struct SegmentationFeaturesRGB segConfRGB={0};
   struct SegmentationFeaturesDepth segConfDepth={0};

   //We want to initialize our segConfRGB for "valid" guessed for our resolution
   initializeRGBSegmentationConfiguration(&segConfRGB,640,480);

   //We want to initialize our segConfDepth for "valid" guessed for our resolution
   initializeDepthSegmentationConfiguration(&segConfDepth,640,480);


   //After the initialization we would like to convert all arguments ( argc,argv ) to the appropriate values
   //on segConfRGB and segConfDepth , this happens inside @ AcquisitionSegment.c
   loadSegmentationDataFromArgs(argc,argv,&segConfRGB,&segConfDepth,&combinationMode);

   //------------------------------------------------------------------
   //------------------------------------------------------------------




  unsigned int i=0;
  for (i=0; i<argc; i++)
  {

     if ( (strcmp(argv[i],"-onlyDepth")==0)||
          (strcmp(argv[i],"-noColor")==0)) {
                                               segConfRGB.saveRGB = 0;
                                           } else
     if ( (strcmp(argv[i],"-onlyColor")==0)||
          (strcmp(argv[i],"-noDepth")==0)) {
                                               segConfDepth.saveDepth = 0;
                                           } else
    if (strcmp(argv[i],"-stereoSplit")==0) {
                                             segConfDepth.saveDepth = 0;
                                             stereoSplit=1;
                                           } else
    if (strcmp(argv[i],"-calibration")==0) {
                                             calibrationSet=1;
                                             strncpy(calibrationFile,argv[i+1],2048);
                                           } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if (strcmp(argv[i],"-module")==0)    {
                                           moduleID_1 = getModuleIdFromModuleName(argv[i+1]);
                                           fprintf(stderr,"Overriding Module Used , set to %s ( %u ) \n",getModuleNameFromModuleID(moduleID_1),moduleID_1);
                                         } else
    if (strcmp(argv[i],"-dev")==0)      {
                                           devID_1 = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID_1);
                                         } else
    if (
         (strcmp(argv[i],"-to")==0) ||
         (strcmp(argv[i],"-o")==0)
        )
        {
          if (strcmp(argv[i+1],"/dev/null")!=0)
          {
           strcpy(outputfoldername,"frames/");
           strcat(outputfoldername,argv[i+1]);
           makepath(outputfoldername);
           fprintf(stderr,"OutputPath , set to %s  \n",outputfoldername);
          } else
          {
           strcpy(outputfoldername,"/dev/null");
          }
         }
       else
    if (strcmp(argv[i],"-seek")==0)      {
                                           seekFrame=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting seek to %u \n",seekFrame);
                                         }
       else
    if (
        (strcmp(argv[i],"-from")==0) ||
        (strcmp(argv[i],"-i")==0)
       )
       { strcat(inputname,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname); }
  }


  if (!acquisitionIsModuleAvailiable(moduleID_1))
   {
       fprintf(stderr,"The module you are trying to use as module A is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS END --------------------------------- */



  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID_1,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module A %s ..\n",getModuleNameFromModuleID(moduleID_1));
       return 1;
   }
   //We want to initialize all possible devices in this example..



  if (segConfRGB.saveRGB==0) { acquisitionDisableStream(moduleID_1,devID_1,0); }
  if (segConfDepth.saveDepth==0) { acquisitionDisableStream(moduleID_1,devID_1,1); }

   char * devName = inputname;
   if (strlen(inputname)<1) { devName=0; }

      if (seekFrame!=0)
      {
          acquisitionSeekFrame(moduleID_1,devID_1,seekFrame);
      }

    //Initialize Every OpenNI Device
    acquisitionOpenDevice(moduleID_1,devID_1,devName,640,480,25);
    acquisitionMapDepthToRGB(moduleID_1,devID_1);


    usleep(1000); // Waiting a while for the glitch frames to pass
    char outfilename[512]={0};


    unsigned int widthRGB , heightRGB , channelsRGB , bitsperpixelRGB;
    acquisitionGetColorFrameDimensions(moduleID_1,devID_1,&widthRGB,&heightRGB ,&channelsRGB , &bitsperpixelRGB );
    if (widthRGB!=segConfRGB.maxX)  { segConfRGB.maxX=widthRGB;  fprintf(stderr,"Updating Segmentation data to match initialized input size Width\n"); }
    if (heightRGB!=segConfRGB.maxY) { segConfRGB.maxY=heightRGB; fprintf(stderr,"Updating Segmentation data to match initialized input size Height\n"); }

    //unsigned char * rgbOut = ( unsigned char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );

    unsigned int widthDepth , heightDepth , channelsDepth , bitsperpixelDepth;
    acquisitionGetDepthFrameDimensions(moduleID_1,devID_1,&widthDepth,&heightDepth ,&channelsDepth , &bitsperpixelDepth );
    if (widthRGB!=segConfDepth.maxX)  { segConfDepth.maxX=widthRGB;  fprintf(stderr,"Updating Segmentation data to match initialized input size Width\n"); }
    if (heightRGB!=segConfDepth.maxY) { segConfDepth.maxY=heightRGB; fprintf(stderr,"Updating Segmentation data to match initialized input size Height\n"); }

    //unsigned short * depthOut = ( unsigned short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );



    if (calibrationSet)
    {
      if (!ReadCalibration(calibrationFile,widthDepth,heightDepth,&calib) )
                                             {
                                               fprintf(stderr,"Could not read calibration file `%s`\n",calibrationFile);
                                               return 1;
                                             }
    } else
    {
      acquisitionGetColorCalibration(moduleID_1,devID_1,&calib);
    }



   while  ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) )
    {
        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID_1,devID_1);


        unsigned long colorTimestamp = acquisitionGetColorTimestamp(moduleID_1,devID_1);
        unsigned char * segmentedRGB = copyRGB(acquisitionGetColorFrame(moduleID_1,devID_1) ,widthRGB , heightRGB);

        unsigned long depthTimestamp = acquisitionGetDepthTimestamp(moduleID_1,devID_1);
        unsigned short * segmentedDepth = copyDepth(acquisitionGetDepthFrame(moduleID_1,devID_1) ,widthDepth , heightDepth);

        if (stereoSplit)
        {
          doStereoSplit(
                         frameNum,
                         segmentedRGB  ,
                         widthRGB,
                         heightRGB,
                         channelsRGB ,
                         bitsperpixelRGB,
                         colorTimestamp,
                         outputfoldername
                        );
        } else
        {
        segmentRGBAndDepthFrame (
                                   segmentedRGB ,
                                   segmentedDepth ,
                                   widthRGB , heightRGB,
                                   &segConfRGB ,
                                   &segConfDepth ,
                                   &calib ,
                                   combinationMode
                                );



        if (strcmp(outfilename,"/dev/null")!=0)
        {
         if (segConfRGB.saveRGB)
         {
          sprintf(outfilename,"%s/colorFrame_%u_%05u.pnm",outputfoldername,devID_1,frameNum);
          acquisitionSimulateTime( colorTimestamp );
          acquisitionSaveRawImageToFile(outfilename,segmentedRGB,widthRGB,heightRGB,channelsRGB,bitsperpixelRGB);
         }

         if (segConfDepth.saveDepth)
         {
          sprintf(outfilename,"%s/depthFrame_%u_%05u.pnm",outputfoldername,devID_1,frameNum);
          acquisitionSimulateTime( depthTimestamp );
          acquisitionSaveRawImageToFile(outfilename,(unsigned char*) segmentedDepth,widthDepth,heightDepth,channelsDepth,bitsperpixelDepth);
         }
        }
       }

       if (segmentedRGB!=0) { free (segmentedRGB); }
       if(segmentedDepth!=0) { free (segmentedDepth); }


       acquisitionStopTimer(0);
       if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
       ++frameNum;
    }

    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);
    closeEverything();
    return 0;
}
