#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "DarknetProcessor.h"
#include "recordOutput.h"

//This is only used for the drawRectangleRGB, if you remove it it is not needed..
#include "../../tools/Drawing/drawing.h"

//if define GPU 1 is not used then nothing will work as it is supposed to be if we are doing a GPU build..
#define GPU 1
#include "../../3dparty/darknet/include/darknet.h"

int mode=0; //0 = detector, 1=classifier
int haveInitialization=0;
int produceFileOutput=1;
int produceRGBOutput=1;



struct darknetDetectorResult
{
 char label[128];
 float probability;
};
#define detectorReturnTopNResults 5
struct darknetDetectorResult detectorResults[detectorReturnTopNResults]={0};





//A structure that holds all the stuff
struct darknetContext
{
 image **alphabet; //This carries the fonts ( https://github.com/pjreddie/darknet/tree/master/data/labels )..
 network * net;    //This is the loaded network where all the magic happens
 float **probs;    //Probabilities of blocks
 char  **names;    //Names of Classes
 detection *dets;  //The new detections structure that carries detections

 float nms;
 float threshold; //Our threshold for what is acceptable
};

//This can be a pointer to a string with a path to a script
//that will get executed every time we have a person detected
char * payload=0;

unsigned int framesProcessed=0;

//Our static darknet context
struct darknetContext dc={0};

int fileExists(char * filename)
{
 if (filename==0) { return 0; }
 //fprintf(stderr,"Checking if file (%s) exists : ",filename);
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */
            fclose(fp);
            //fprintf(stderr,"yes\n");
            return 1;
          }
 /* doesnt exist */
 //fprintf(stderr,"no\n");
 return 0;
}

unsigned int countLinesOfFile(const char * path)
{
  FILE *fp = fopen(path,"r");
  if (fp!=0)
  {
    unsigned int lines=0;
    while(!feof(fp))
     {
      char ch = fgetc(fp);
      if(ch == '\n')
      {
       lines++;
      }
     }
    fclose(fp);
    return lines;
  }
  return 0;
}

//This image converts a raw pointer to an image structure as needed by stb_image
image convertBufferToImage(void * pixels , unsigned int width, unsigned int height, unsigned int channels)
{
    unsigned char * data = (unsigned char*) pixels;
    int w=width, h=height, c=channels , step=width*channels;

    image im = make_image(w, h, c);

    int i,j,k;
    for(i = 0; i < h; ++i)
    {
      for(k= 0; k < c; ++k)
        {
          for(j = 0; j < w; ++j)
            {
                im.data[k*w*h + i*w + j] = (float) data[i*step + j*c + k]/255.1; //Prevent saturation
            }
        }
    }
    return im;
}

int internalDarknetInitialization(
                                  const char *cfgfile,
                                  const char *weightfile,
                                  const char *datafile,
                                  const char *namelistfile,
                                  float threshold,
                                  float nms
                                 )
{
    if ( (!cfgfile) || (!weightfile) )  { return 0; }

    //Random Seed
    srand(time(NULL));

    fprintf(stderr,"--------------- Darknet Configuration ---------------\n");
    fprintf(stderr,"Data Source : %s\n",datafile);
    fprintf(stderr,"Names list Source : %s\n",namelistfile);
    fprintf(stderr,"CFG Source : %s\n",cfgfile);
    fprintf(stderr,"Weight Source : %s\n",weightfile);
    fprintf(stderr,"Threshold : %0.2f\n",threshold);
    fprintf(stderr,"NMS : %0.2f\n",nms);

    int classes =0;
    //We can auto count number of classes from the labels file..
    if (namelistfile!=0)
    {
      classes = countLinesOfFile(namelistfile);
    }

    //We can also parse the data file to load settings..
    if (datafile!=0)
    {
     fprintf(stderr,"Data file `%s` is going to override parameters..",datafile);
     list *options   = read_data_cfg(datafile);
     if (options!=0)
     {
      classes     = option_find_int(options, "classes", 20);
      namelistfile = option_find_str(options, "names", namelistfile);
      fprintf(stderr,"Names list Source changed to : %s\n",namelistfile);
     }
    }
    fprintf(stderr,"Number of Classes : %u\n",classes);

    if (classes==0)
    {
      fprintf(stderr,"Operating with 0 classes failed to correctly load settings..\n");
      return 0;
    }


    dc.names        = get_labels(namelistfile);
    dc.nms=nms;
    dc.threshold=threshold;

    dc.alphabet = load_alphabet();

    dc.net = load_network(cfgfile, weightfile, 0);
    set_batch_network(dc.net, 1);

    //fprintf(stderr,"Retreiving last layer\n");
    layer l = dc.net->layers[dc.net->n-1];

    fprintf(stderr,"Images will be resized to fit %ux%u \n",dc.net->w, dc.net->h);
    fprintf(stderr,"----------------------------------------------------\n");


    fprintf(stderr,"Allocating space ..\n");
    dc.probs = (float **)calloc( l.w * l.h * l.n, sizeof(float *));

    int j;
    for(j = 0; j < l.w * l.h * l.n; ++j)
         { dc.probs[j] = (float *)calloc( l.classes+1, sizeof(float)); }

    fprintf(stderr,"Done with initialization ..\n");
    haveInitialization=1;
 return 1;
}

int initArgs_DarknetProcessor(int argc, char *argv[])
{
 char * cfgFile=0;
 char * weightFile=0;
 char * dataFile=0;
 char * namelistFile=0;

 float threshold=0.55; // was 0.35
 float nms=0.4;


 unsigned int i=0;
 for (i=0; i<argc; i++)
 {
   if (strstr(argv[i],".cfg")!=0)        { cfgFile=argv[i]; } else
   if (strstr(argv[i],".weights")!=0)    { weightFile=argv[i]; } else
   if (strstr(argv[i],".data")!=0)       { dataFile=argv[i]; } else
   if (
        (strstr(argv[i],".names")!=0) ||
        (strstr(argv[i],"labels.txt")!=0)
       )
                                          { namelistFile=argv[i]; } else
   if (strstr(argv[i],"--payload")!=0)         { payload=argv[i+1]; } else
   if (strstr(argv[i],"--threshold")!=0)       { threshold=atof(argv[i+1]); } else
   if (strstr(argv[i],"--nms")!=0)             { nms=atof(argv[i+1]); } else
   if (strstr(argv[i],"--detector")!=0)        { mode=0; } else
   if (strstr(argv[i],"--classifier")!=0)      { mode=1; } else
   if (strstr(argv[i],"--noVisualization")!=0) { produceRGBOutput=0; } else
   if (strstr(argv[i],"--noFileOutput")!=0)    { produceFileOutput=0; } else
   if (strstr(argv[i],"--fast")!=0)            { produceFileOutput=0; produceRGBOutput=0; }


 }

 #if GPU
  fprintf(stderr,"Thinking about GPUS\n");
  signed int gpu_index = find_int_arg(argc, argv, "-gpuid", 0);
  if(find_arg(argc, argv, "-nogpu"))
    {
        gpu_index = -1;
    }

   if(gpu_index >= 0) {
                        fprintf(stderr,"Setting CUDA device ( gpu_index=%d )..\n",gpu_index);
                        cuda_set_device(gpu_index);
                      } else
                      {
                        fprintf(stderr,"Running without GPU ( gpu_index=%d )..\n",gpu_index);
                      }
 #else
   fprintf(stderr,"Processor is compiled without GPU define, if Darknet is compiled WITH it you may experience crazy problems since ");
   fprintf(stderr,"Darknet ABI changes depending on this flag..!\n");
 #endif // GPU

 framesProcessed=resumeFrameOutput();
 fprintf(stderr,"Resuming @ %u\n",framesProcessed);

 return internalDarknetInitialization(
                                       cfgFile,    //"yourpath/yolo.cfg"
                                       weightFile, //"yourpath/yolo.weights"
                                       dataFile,   //"yourpath/coco.data"
                                       namelistFile,//"data/names.list"
                                       threshold,
                                       nms
                                     );
}




int receiveDetection(
                       //This is the image we will draw to return back to output
                       void * inputPixels, unsigned int inputWidth, unsigned int inputHeight,

                       //This is the image internally used by the neural network
                       image * im,

                       //Detection Details
                       int   classification ,
                       float probability,
                       float x,
                       float y,
                       float detectionWidth,
                       float detectionHeight,

                       //System/Logging Stuff
                       FILE * logOutputFile,
                       char * directoryToUseForOutput,
                       struct tm * currentTime,
                       unsigned int frameNumber
                    )
{
  if (x>1.0) { return 0; }
  if (y>1.0) { return 0; }

  if (probability >  dc.threshold)
   {
      //Calculate coordinates from floats..
      float halfWidth  = (float) detectionWidth/2;
      float halfHeight = (float) detectionHeight/2;

      float x1 = (x-halfWidth)  *inputWidth;
      float y1 = (y-halfHeight) *inputHeight;
      float x2 = (x+halfWidth)  *inputWidth;
      float y2 = (y+halfHeight) *inputHeight;

      if (x1<0) { x1=0; }
      if (y1<0) { y1=0; }
      if (x2>inputWidth)  { x2=inputWidth;  }
      if (y2>inputHeight) { y2=inputHeight; }


      //This uses my own drawing "../../tools/Drawing/drawing.h" methods , you might want to comment out
      if (produceRGBOutput)
      {
       drawRectangleRGB(inputPixels,inputWidth,inputHeight, 255,0,0,  3, (unsigned int) x1 , (unsigned int) y1 , (unsigned int) x2 , (unsigned int) y2 );
      }


      //This outputs detection to stderr
      fprintf(stderr,"%s-> %f%% @ %f %f %f %f\n", dc.names[classification], probability*100, x1, y1, detectionWidth, detectionHeight);


      //This outputs detection to logfile
      if (produceFileOutput)
      {
               logEvent(
                        logOutputFile,
                        currentTime,
                        framesProcessed,
                        x1,
                        y1,
                        detectionWidth,
                        detectionHeight,
                        dc.names[classification],
                        probability*100
                      );

      }


      //If we this is a person we want to save the image / execute scripts etc..
       if (strcmp(dc.names[classification],"person")==0)
                  {
                   if (detectionHeight > 0.3 )
                     {
                      if (payload!=0)
                      {
                        int i=system(payload);
                        if (i!=0) { fprintf(stderr,"Payload (%s) failed..\n",payload); }
                      }
                       //if it is a standing human..
                       if (produceFileOutput)
                       {
                        char recordFile[512]={0};
                        snprintf(recordFile,512,"%s/record_%u",directoryToUseForOutput,frameNumber);
                        save_image(*im,recordFile);
                       }
                     }
                  }

     return 1;
   }
 return 0;
}







int doDetector(image * im, layer * l ,  void * data, unsigned int width, unsigned int height)
{
  //We got back something which needs to be decoded

  //This will hold the number of boxes
     int nboxes = 0;

     //We get back the detections as per the YOLO paper
     dc.dets = get_network_boxes(dc.net, 1, 1, dc.threshold, 0, 0, 0, &nboxes);

     //The detections are sorted and duplicates discarded
     if (dc.nms) { do_nms_sort(dc.dets, l->w*l->h*l->n, l->classes, dc.nms); }

     //Detections are detected..! :P
     get_detection_detections(*l, 1 , 1 , dc.threshold, dc.dets);


     //This is just to log time..
     time_t clock = time(NULL);
     struct tm * currentTime = gmtime ( &clock );


     //Rectangles, labels etc are added to im
     int num = l->side*l->side*l->n;
     draw_detections(*im, dc.dets,num, dc.threshold, dc.names, dc.alphabet, l->classes);


     //This is the directory we want to dump output ( plus the date )
     char directoryToUse[1024];
     snprintf(directoryToUse,1024,"%u_%02u_%02u", EPOCH_YEAR_IN_TM_YEAR+currentTime->tm_year, currentTime->tm_mon+1, currentTime->tm_mday);
     useLoggingDirectory(directoryToUse);


     //This surveilance.log will be appended with the last results
     char logFile[1024];
     snprintf(logFile,1024,"%s/surveilance.log",directoryToUse);

     FILE * fp = startLogging(logFile);

     //Go Through all detections
     unsigned int i=0,j=0;
     for(i = 0; i < nboxes; ++i)
      {
        for(j = 0; j < l->classes; ++j)
        {
            //If probability is not zero
            if (dc.dets[i].prob[j])
            {
              //This detection is important and we need to consider it
              receiveDetection(
                              data, width, height ,
                              im,
                              //Detection Stuff
                              j,                  //Class
                              dc.dets[i].prob[j], //Probability
                              dc.dets[i].bbox.x,  //X
                              dc.dets[i].bbox.y,  //Y
                              dc.dets[i].bbox.w,  //Width
                              dc.dets[i].bbox.h,  //Height
                              //System stuff
                              fp,
                              directoryToUse,
                              currentTime,
                              framesProcessed
                            );
            }
        }
    }

    //Free Detections
    free_detections(dc.dets,nboxes);

    //Done with current frame
    fflush(fp);
    stopLogging(fp);

    //If we have OpenCV enabled in Darknet build this will output a window
    //show_image(*im, "predictions");



}



int doClassifier(image * im, layer * l ,  float *predictions, void * data, unsigned int width, unsigned int height)
{
 #ifdef GPU
        cuda_pull_array(l->output_gpu, l->output, l->outputs);
 #endif
        unsigned int i;
        /*
           for(i = 0; i < l->outputs; ++i)
           {
            printf("%f\n", l->output[i]);
           }
           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
       */
        int *indexes = calloc(detectorReturnTopNResults, sizeof(int));

        top_predictions(dc.net, detectorReturnTopNResults, indexes);
        printf("Predicted \n");
        for(i = 0; i < detectorReturnTopNResults; ++i)
        {
            int index = indexes[i];
            printf("%s: %f\n", dc.names[index], predictions[index]);
            snprintf(detectorResults[i].label,128,"%s",dc.names[index]);
            detectorResults[i].probability = predictions[index];
        }

        free(indexes);
}



char * getDetectionLabel_DarknetProcessor(unsigned int detectionNumber)
{
  if (detectionNumber>=detectorReturnTopNResults) { return 0; }
  return detectorResults[detectionNumber].label;
}


float getDetectionProbability_DarknetProcessor(unsigned int detectionNumber)
{
  if (detectionNumber>=detectorReturnTopNResults) { return 0.0; }
  return detectorResults[detectionNumber].probability;
}





int addDataInput_DarknetProcessor(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
 if (!haveInitialization)
    {
      fprintf(stderr,"Darknet Processor was not initialized properly and will not do anything..\n");
      return 0;
    }

 //fprintf(stderr,"addDataInput_DarknetProcessor %u (%ux%u) channels=%u\n" , stream , width, height,channels);
 if (stream==0)
 {
    //This is the original input image that was given to addDataInput_DarknetProcessor
    image im=convertBufferToImage(data, width, height, channels);


    //We might want to resize the image to make it 448x448
    image sized = resize_image(im, dc.net->w, dc.net->h);

    //But another idea could be to letterbox it to 448x448 , which is disabled
    //image sized = letterbox_image(im, dc.net->w, dc.net->h);

    //We can also dump it to disk to view it , also disabled
    //save_image(sized, "sized");


    //This is the final layer of the net
    layer l = dc.net->layers[dc.net->n-1];

    //Detecting happens here
    float *prediction = network_predict(dc.net /*Neural Net*/, sized.data /*Search Image*/);


    if (prediction!=0)
    {
      switch (mode)
      {
        case 0: doDetector(&im,&l,data,width,height); break;
        case 1: doClassifier(&im,&l,prediction,data,width,height); break;
      }
    } else
    {
     fprintf(stderr,"Failed to run network ( prediction points to %p )..\n",prediction);
    }

    //Free memory to ensure no leaks..
    free_image(im);
    free_image(sized);

    ++framesProcessed;
  return 1;
 }
 return 0;
}


























int stop_DarknetProcessor()
{
    int j;
    layer l = dc.net->layers[dc.net->n-1];

    if (dc.probs!=0)
    {
     for(j = 0; j < l.w * l.h * l.n; ++j)
         { free(dc.probs[j]); }

     free(dc.probs);
     dc.probs =0;
    }


    free_network(dc.net);

    //TODO: dealloc these
    //image **alphabet; //This carries the fonts ( https://github.com/pjreddie/darknet/tree/master/data/labels )..
    //char  **names;

  return 0;
}



unsigned char * getDataOutput_DarknetProcessor(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
  return 0;
}


// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// The rest of the processor interface calls are not implemented so they are just included here as a reminder .. :P
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
int setConfigStr_DarknetProcessor(char * label,char * value) { return 0; }
int setConfigInt_DarknetProcessor(char * label,int value)    { return 0; }
int processData_DarknetProcessor() { return 0; }
int cleanup_DarknetProcessor()     { return 0; }
unsigned short * getDepth_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)  { return 0; }
unsigned char * getColor_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)   { return 0; }
