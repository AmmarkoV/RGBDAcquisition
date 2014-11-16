#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../acquisition/Acquisition.h"
#include "../acquisition_mux/AcquisitionMux.h"

char outputfoldername[512]={0};

int enforceCalibrationSharing = 1;
char inputname1[512]={0};
char inputname2[512]={0};


unsigned int defaultWidth=640;
unsigned int defaultHeight=480;
unsigned int transparency=0;
unsigned char transR=0,transG=0,transB=0;
signed int shiftX=0;
signed int shiftY=0;
signed int shiftTime=0;

unsigned int frameDoublerEnabled=0;

int calibrationSetA = 0;
struct calibration calibA;
int calibrationSetB = 0;
struct calibration calibB;

unsigned int longExposure = 0;
unsigned long * rgbCollector =0;
unsigned long * depthCollector =0;

unsigned int saveColor=1 , saveDepth=1;
//We want to grab multiple frames in this example if the user doesnt supply a parameter default is 10..
unsigned int frameNum=0,maxFramesToGrab=10;

unsigned int devID_1=0 , devID_2=0;
ModuleIdentifier moduleID_1 = OPENGL_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//
ModuleIdentifier moduleID_2 = TEMPLATE_ACQUISITION_MODULE;//OPENNI1_ACQUISITION_MODULE;//



void closeEverything()
{
 fprintf(stderr,"Gracefully closing everything .. ");
    acquisitionCloseDevice(moduleID_1,devID_1);
    acquisitionCloseDevice(moduleID_2,devID_2);
    acquisitionStopModule(moduleID_1);
    acquisitionStopModule(moduleID_2);
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

  strcpy(outputfoldername,"frames/");

  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS --------------------------------- */

  int i=0;
  for (i=0; i<argc; i++)
  {
     if (strcmp(argv[i],"-longExposure")==0)
     {
         longExposure=1;
         fprintf(stderr,"Long Exposure mode initialized\n");
     } else
     if ( (strcmp(argv[i],"-onlyDepth")==0)||
          (strcmp(argv[i],"-noColor")==0)) {
                                               saveColor = 0;
                                           } else
     if ( (strcmp(argv[i],"-onlyColor")==0)||
          (strcmp(argv[i],"-noDepth")==0)) {
                                               saveDepth = 0;
                                           } else

    if (strcmp(argv[i],"-calibration1")==0) {
                                             calibrationSetA=1;
                                             if (!ReadCalibration(argv[i+1],defaultWidth,defaultHeight,&calibA) )
                                             {
                                               fprintf(stderr,"Could not read calibration file for file 1 `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-calibration2")==0) {
                                             calibrationSetB=1;
                                             if (!ReadCalibration(argv[i+1],defaultWidth,defaultHeight,&calibB) )
                                             {
                                               fprintf(stderr,"Could not read calibration file for file 2 `%s`\n",argv[i+1]);
                                               return 1;
                                             }
                                           } else
    if (strcmp(argv[i],"-maxFrames")==0) {
                                           maxFramesToGrab=atoi(argv[i+1]);
                                           fprintf(stderr,"Setting frame grab to %u \n",maxFramesToGrab);
                                         } else
    if  (
          (strcmp(argv[i],"-transparency")==0) ||
          (strcmp(argv[i],"-trans")==0)
        )
                                         {
                                           transparency=atoi(argv[i+1]);
                                           if (transparency>100) { transparency=100; }
                                           //Transparency 0 = no transparency , transparency 100 = full transparency
                                           fprintf(stderr,"Setting transparency to %u \n",transparency);
                                         } else
    if (strcmp(argv[i],"-module1")==0)    {
                                           moduleID_1 = getModuleIdFromModuleName(argv[i+1]); // //Module 1 is Base
                                           fprintf(stderr,"Overriding Module 1 Used , set to %s ( %u ) \n",getModuleNameFromModuleID(moduleID_1),moduleID_1);
                                         } else
    if (strcmp(argv[i],"-module2")==0)   {
                                           moduleID_2 = getModuleIdFromModuleName(argv[i+1]); //Module 2 is Overlay
                                           fprintf(stderr,"Overriding Module 2 Used , set to %s ( %u ) \n",getModuleNameFromModuleID(moduleID_2),moduleID_2);
                                         } else
    if (strcmp(argv[i],"-dev1")==0)      {
                                           devID_1 = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID_1);
                                         } else
    if (strcmp(argv[i],"-dev2")==0)      {
                                           devID_2 = atoi(argv[i+1]);
                                           fprintf(stderr,"Overriding device Used , set to %s ( %u ) \n",argv[i+1],devID_2);
                                         } else
    if ((strcmp(argv[i],"-overlayBackground")==0) ||
        (strcmp(argv[i],"-background")==0) )  {
                                                  transR = atoi(argv[i+1]);
                                                  transG = atoi(argv[i+2]);
                                                  transB = atoi(argv[i+3]);
                                                  fprintf(stderr,"Setting OverlayBackground RGB(%u,%u,%u)\n",transR,transG,transB);
                                              } else
    if (strcmp(argv[i],"-shiftX")==0)      {
                                            shiftX = atoi(argv[i+1]);
                                            fprintf(stderr,"Adding a horizontal shift ( %u ) \n",argv[i+1],shiftX);
                                          } else
    if (strcmp(argv[i],"-shiftY")==0)     {
                                            shiftY = atoi(argv[i+1]);
                                            fprintf(stderr,"Adding a horizontal shift ( %u ) \n",argv[i+1],shiftY);
                                          } else
    if (strcmp(argv[i],"-shiftTime")==0)    {
                                              shiftTime = atoi(argv[i+1]);
                                              fprintf(stderr,"Adding a time shift ( %u ) \n",argv[i+1],shiftTime);
                                            } else
    if (
        (strcmp(argv[i],"-frameDoubler")==0)
       )
       { frameDoublerEnabled=1; fprintf(stderr,"Frame Doubler Will be engaged \n"); }
      else

    if (
        (strcmp(argv[i],"-from1")==0) ||
        (strcmp(argv[i],"-i1")==0)||
        (strcmp(argv[i],"-fromBase")==0)
       )
       { strcat(inputname1,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname1); }
      else
    if (
         (strcmp(argv[i],"-from2")==0)||
         (strcmp(argv[i],"-i2")==0)||
        (strcmp(argv[i],"-fromOverlay")==0)
       )
       { strcat(inputname2,argv[i+1]); fprintf(stderr,"Input , set to %s  \n",inputname2); }
      else
    if (
        (strcmp(argv[i],"-o")==0) ||
        (strcmp(argv[i],"-to")==0)
       )
                                         {
                                           strcpy(outputfoldername,"frames/");
                                           strcat(outputfoldername,argv[i+1]);
                                           makepath(outputfoldername);
                                           fprintf(stderr,"OutputPath , set to %s  \n",outputfoldername);
                                         }
  }


  if (!acquisitionIsModuleAvailiable(moduleID_1))
   {
       fprintf(stderr,"The module you are trying to use as module A is not linked in this build of the Acquisition library..\n");
       return 1;
   }

  if (!acquisitionIsModuleAvailiable(moduleID_2))
   {
       fprintf(stderr,"The module you are trying to use as module B is not linked in this build of the Acquisition library..\n");
       return 1;
   }
  /*! --------------------------------- INITIALIZATION FROM COMMAND LINE PARAMETERS END --------------------------------- */





  //We need to initialize our module before calling any related calls to the specific module..
  if (!acquisitionStartModule(moduleID_1,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module A %s ..\n",getModuleNameFromModuleID(moduleID_1));
       return 1;
   }
  if (!acquisitionStartModule(moduleID_2,16 /*maxDevices*/ , 0 ))
  {
       fprintf(stderr,"Could not start module B %s ..\n",getModuleNameFromModuleID(moduleID_2));
       return 1;
   }


  //We want to initialize all possible devices in this example..
  if (moduleID_1==moduleID_2) { ++devID_2; }

   if ( calibrationSetA )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",calibA.farPlane,calibA.nearPlane);
    acquisitionSetColorCalibration(moduleID_1,devID_1,&calibA);
    acquisitionSetDepthCalibration(moduleID_1,devID_1,&calibA);
   }

   if ( calibrationSetB )
   {
    fprintf(stderr,"Set Far/Near to %f/%f\n",calibB.farPlane,calibB.nearPlane);
    acquisitionSetColorCalibration(moduleID_2,devID_2,&calibB);
    acquisitionSetDepthCalibration(moduleID_2,devID_2,&calibB);
   }


  char * devName1 = inputname1;
  if (strlen(inputname1)<1) { devName1=0; }
  char * devName2 = inputname2;
  if (strlen(inputname2)<1) { devName2=0; }

    //Initialize Every OpenNI Device
    acquisitionOpenDevice(moduleID_1,devID_1,devName1,defaultWidth,defaultHeight,25);
    acquisitionMapDepthToRGB(moduleID_1,devID_1);

    acquisitionOpenDevice(moduleID_2,devID_2,devName2,defaultWidth,defaultHeight,25);
    acquisitionMapDepthToRGB(moduleID_2,devID_2);


    if ( enforceCalibrationSharing )
    {
      struct calibration calib={0};
      acquisitionGetColorCalibration(moduleID_1,devID_1,&calib);
      acquisitionSetColorCalibration(moduleID_2,devID_2,&calib);
      acquisitionGetDepthCalibration(moduleID_1,devID_1,&calib);
      acquisitionSetDepthCalibration(moduleID_2,devID_2,&calib);
    }

    usleep(1000); // Waiting a while for the glitch frames to pass
    char outfilename[512]={0};


    unsigned int widthRGB , heightRGB , channelsRGB , bitsperpixelRGB;
    acquisitionGetColorFrameDimensions(moduleID_1,devID_1,&widthRGB,&heightRGB ,&channelsRGB , &bitsperpixelRGB );
    //Todo , check with module 2 bla bla
    unsigned char * rgbOut = (unsigned char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );

    unsigned int widthDepth , heightDepth , channelsDepth , bitsperpixelDepth;
    acquisitionGetDepthFrameDimensions(moduleID_1,devID_1,&widthDepth,&heightDepth ,&channelsDepth , &bitsperpixelDepth );
    unsigned short * depthOut = (unsigned short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );


    unsigned char * doubleRGB = 0 , doubleRGBOut = 0;
    unsigned short * doubleDepth = 0 , doubleDepthOut = 0;
    if (frameDoublerEnabled)
    {
      doubleRGB = (unsigned char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );
      doubleRGBOut = (unsigned char* )  malloc(widthRGB*heightRGB*channelsRGB * (bitsperpixelRGB/8 ) );
      doubleDepth = (unsigned short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );
      doubleDepthOut = (unsigned short* )  malloc(widthDepth*heightDepth*channelsDepth * (bitsperpixelDepth/8 ) );
    }


    if (longExposure)
    {
      rgbCollector = (unsigned long * ) malloc(sizeof(unsigned long) * widthRGB * heightRGB*3 );
      memset(rgbCollector,0,sizeof(unsigned long) * widthRGB * heightRGB*3 );
      depthCollector = (unsigned long * ) malloc(sizeof(unsigned long) * widthRGB * heightRGB );
      memset(depthCollector,0,sizeof(unsigned long) * widthRGB * heightRGB);
    }

    fprintf(stderr,"Base Module is %s , device %u , Overlay Module is %s , device %u\n",getModuleNameFromModuleID(moduleID_1),devID_1, getModuleNameFromModuleID(moduleID_2),devID_2);
    if (shiftTime==0) { /* No time shift */ } else
    if (shiftTime>0) { for (frameNum=0; frameNum<abs(shiftTime); frameNum++) { acquisitionSnapFrames(moduleID_2,devID_2); } } else
    if (shiftTime<0) { for (frameNum=0; frameNum<abs(shiftTime); frameNum++) { acquisitionSnapFrames(moduleID_1,devID_1); } }

   while  ( (maxFramesToGrab==0)||(frameNum<maxFramesToGrab) )
    {

        if (maxFramesToGrab!=0)
          { fprintf(stderr,"Muxing %0.2f %% , grabbed frame %u/%u \n", (float) frameNum*100/maxFramesToGrab , frameNum,maxFramesToGrab); } else
          { fprintf(stderr,"Muxing an unknown total number of frames , currently at %u \n",frameNum); }

        acquisitionStartTimer(0);

        acquisitionSnapFrames(moduleID_1,devID_1);
        acquisitionSnapFrames(moduleID_2,devID_2);

        if (longExposure)
        {
          LongExposureFramesCollect( acquisitionGetColorFrame(moduleID_1,devID_1) , rgbCollector ,
                                     acquisitionGetDepthFrame(moduleID_1,devID_1),  depthCollector ,
                                     widthRGB , heightRGB, &frameNum );
        } else
        {
        mux2RGBAndDepthFrames
          (
           acquisitionGetColorFrame(moduleID_1,devID_1) , //Module 1 is Base
           acquisitionGetColorFrame(moduleID_2,devID_2) , //Module 2 is Overlay
           rgbOut ,
           acquisitionGetDepthFrame(moduleID_1,devID_1) ,
           acquisitionGetDepthFrame(moduleID_2,devID_2) ,
           depthOut ,
           transR,transG,transB,
           shiftX,shiftY,
           widthRGB , heightRGB ,
           transparency , 0 );


         if (saveColor)
          {
          sprintf(outfilename,"%s/colorFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,rgbOut,widthRGB , heightRGB, channelsRGB , bitsperpixelRGB);
          }

         if (saveDepth)
         {
          sprintf(outfilename,"%s/depthFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,(unsigned char*) depthOut,widthDepth , heightDepth, channelsDepth , bitsperpixelDepth);
         }
        }

       if (frameDoublerEnabled)
       {
        if (frameNum!=0)
        {
            /*
          generateInterpolatedFrames(
                                      doubleRGB, rgbOut , doubleRGBOut ,
                                      doubleDepth, depthOut , doubleDepthOut ,
                                      widthRGB , heightRGB
                                     );*/
          ++frameNum;

         if (saveColor)
          {
          sprintf(outfilename,"%s/colorFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,doubleRGBOut,widthRGB , heightRGB, channelsRGB , bitsperpixelRGB);
          }

         if (saveDepth)
         {
          sprintf(outfilename,"%s/depthFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,(unsigned char*) doubleDepthOut,widthDepth , heightDepth, channelsDepth , bitsperpixelDepth);
         }
        }

         memcpy( doubleRGB , rgbOut , sizeof(unsigned char) * widthRGB * heightRGB*3 );
         memcpy( doubleDepth , depthOut , sizeof(unsigned long) * widthRGB * heightRGB );
       }




       acquisitionStopTimer(0);
       if (frameNum%25==0) fprintf(stderr,"%0.2f fps\n",acquisitionGetTimerFPS(0));
       ++frameNum;
    }


    if (longExposure)
    {
      LongExposureFramesFinalize(rgbCollector ,  rgbOut ,
                                 depthCollector ,  depthOut,
                                 widthRGB , heightRGB , &frameNum);

      free(rgbCollector);
      free(depthCollector);
         if (saveColor)
          {
          sprintf(outfilename,"%s/colorFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,rgbOut,widthRGB , heightRGB, channelsRGB , bitsperpixelRGB);
          }

         if (saveDepth)
         {
          sprintf(outfilename,"%s/depthFrame_%u_%05u",outputfoldername,devID_1,frameNum);
          saveMuxImageToFile(outfilename,(unsigned char*) depthOut,widthDepth , heightDepth, channelsDepth , bitsperpixelDepth);
         }
    }


    fprintf(stderr,"Done grabbing %u frames! \n",maxFramesToGrab);


    //-----------------------------------------------
    if (depthOut != 0) { free(depthOut); }
    if (rgbOut != 0) { free(rgbOut); }
    //-----------------------------------------------
    if (doubleRGB != 0) { free(doubleRGB); }
    if (doubleRGBOut != 0) { free(doubleRGBOut); }
    //-----------------------------------------------
    if (doubleDepth != 0) { free(doubleDepth); }
    if (doubleDepthOut != 0) { free(doubleDepthOut); }
    //-----------------------------------------------


    closeEverything();


    return 0;
}


