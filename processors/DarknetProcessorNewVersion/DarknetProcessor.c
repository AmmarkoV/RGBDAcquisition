#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "DarknetProcessor.h"
#include "recordOutput.h"

#include "../../tools/Drawing/drawing.h"

#define GPU 1 //if define GPU1 is not used then nothing will work as it is supposed to be if we are doing a GPU build..
//#define OPENCV 1
//#define CUDNN 1


#include "../../3dparty/darknet/include/darknet.h"

unsigned int framesProcessed=0;

struct darknetContext
{
 image **alphabet;
 network * net;

 float nms;

 char **names;

 detection *dets;

 float threshold;
 float hierarchyThreshold;
};

char * payload=0;
struct darknetContext dc={0};

image load_image_from_buffer(void * pixels , unsigned int width, unsigned int height, unsigned int channels)
{
    unsigned char * data = (unsigned char*) pixels;
    int w=width, h=height, c=channels , step=width*channels;
    int i,j,k;
    image im = make_image(w, h, c);


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

    //free(data);
    return im;
}

int init_yolo(
                const char *cfgfile,
                const char *weightfile,
                const char *datafile,
                float thresh
               )
{
    if (!cfgfile)    { return 0; }
    if (!weightfile) { return 0; }
    if (!datafile) { return 0; }

    dc.alphabet = load_alphabet();

    fprintf(stderr,"Data Source : %s\n",datafile);
    fprintf(stderr,"CFG Source : %s\n",cfgfile);
    fprintf(stderr,"Weight Source : %s\n",weightfile);

    list *options = read_data_cfg(datafile);
    int classes = option_find_int(options, "classes", 20);
    char *name_list = option_find_str(options, "names", "data/names.list");
    dc.names = get_labels(name_list);

    dc.net = load_network(cfgfile, weightfile, 0);
    set_batch_network(dc.net, 1);

    srand(2222222);

    fprintf(stderr,"Retreiving last layer\n");
    layer l = dc.net->layers[dc.net->n-1];
    dc.nms=.4;
    dc.threshold=thresh;
    dc.hierarchyThreshold=0.5;

    fprintf(stderr,"Done with initialization ..\n");
 return 1;
}

int initArgs_DarknetProcessor(int argc, char *argv[])
{
 char * cfgFile=0;
 char * weightFile=0;
 char * dataFile=0;
 float threshold=0.35; // was 0.35


 unsigned int i=0;
 for (i=0; i<argc; i++)
 {
   if (strstr(argv[i],".cfg")!=0)      { cfgFile=argv[i]; }
   if (strstr(argv[i],".weights")!=0)  { weightFile=argv[i]; }
   if (strstr(argv[i],".data")!=0)     { dataFile=argv[i]; }
   if (strstr(argv[i],"--payload")!=0) { payload=argv[i+1]; }
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
 #endif // GPU

 framesProcessed=resumeFrameOutput();
 fprintf(stderr,"Resuming @ %u\n",framesProcessed);

 return init_yolo(
                   cfgFile,
                   weightFile,
                   dataFile,
                   threshold
                 );
}


int addDataInput_DarknetProcessor(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
 //fprintf(stderr,"addDataInput_DarknetProcessor %u (%ux%u) channels=%u\n" , stream , width, height,channels);
 if (stream==0)
 {
    image im=load_image_from_buffer(data, width, height, channels);
    //save_image(im, "original");

    //image sized = resize_image(im, dc.net->w, dc.net->h);
    image sized = letterbox_image(im, dc.net->w, dc.net->h);
    //save_image(sized, "sized");

    layer l = dc.net->layers[dc.net->n-1];

    //fprintf(stderr,"detecting.. ");

    float *X = sized.data;
    float *prediction = network_predict(dc.net /*Neural Net*/, X /*Search Image*/);
    //fprintf(stderr,"done ( %u )\n",l.outputs);


    //if (l.outputs!=0)
    {
     int nboxes = 0;
     dc.dets = get_network_boxes(dc.net, 1, 1, dc.threshold, 0, 0, 0, &nboxes);
     if (dc.nms) { do_nms_sort(dc.dets, l.w*l.h*l.n, l.classes, dc.nms); }


    //printf("Objects (%u classes):\n\n",l.classes);

    unsigned int detections =  l.w * l.h * l.n;


   time_t clock = time(NULL);
   struct tm * ptm = gmtime ( &clock );

    int num = l.side*l.side*l.n;
    draw_detections(im, dc.dets,num, dc.threshold, dc.names, dc.alphabet, l.classes);

    //save_image(im, "predictions");
    //show_image(im, "predictions");


    char directoryToUse[1024];
    snprintf(directoryToUse,1024,"%u_%02u_%02u", EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year, ptm->tm_mon+1, ptm->tm_mday);
    useLoggingDirectory(directoryToUse);


    char logFile[1024];
    snprintf(logFile,1024,"%s/surveilance.log",directoryToUse);

    FILE * fp = startLogging(logFile);








    int i,j;

    for(i = 0; i < num; ++i)
    {
        char labelstr[4096] = {0};
        int detectionclass = -1;
        for(j = 0; j < l.classes; ++j){
            if (dc.dets[i].prob[j] > dc.threshold){
                if (detectionclass  < 0) {
                    strcat(labelstr, dc.names[j]);
                    detectionclass  = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, dc.names[j]);
                }
                printf("%s: %.0f%%\n", dc.names[j], dc.dets[i].prob[j]*100);
            }
        }
        if(detectionclass  >= 0)
           {
            int width = im.h * .006;

            box b = dc.dets[i].bbox;
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

             if (dc.dets[i].prob[j] >  dc.threshold)
            {
               logEvent(
                        fp,
                        ptm,
                        framesProcessed,
                        b.x,
                        b.y,
                        b.w,
                        b.h,
                        dc.names[j],
                        dc.dets[i].prob[j]*100
                      );


               if (strcmp(dc.names[j],"person")==0)
                  {
                   //fprintf(stderr,"%0.2f\n",dc.boxes[i].h);
                   if ( b.h > 0.3 )
                     {
                      if (payload!=0)
                      {
                        int i=system(payload);
                        if (i!=0) { fprintf(stderr,"Payload (%s) failed..\n",payload); }
                      }
                       //if it is a standing human..
                        char recordFile[512]={0};
                        snprintf(recordFile,512,"%s/record_%u",directoryToUse,framesProcessed);
                        save_image(im,recordFile);
                     }
                  }


              drawRectangleRGB(data,width,height, 255,0,0,  3, left , top , right , bot );

            }

            }
    }












    for(i = 0; i <detections; ++i)
    {
        for(j = 0; j < l.classes; ++j){

        }
    } // End for loop
    fflush(fp);
    stopLogging(fp);

    //show_image(im, "predictions");
    }
    /*
    else
    {
     fprintf(stderr,"Failed to run network ( prediction points to %p )..\n",prediction);
    }*/

    free_image(im);
    free_image(sized);

   ++framesProcessed;
  return 1;
 }
 return 0;
}

int setConfigStr_DarknetProcessor(char * label,char * value)
{
 return 0;

}

int setConfigInt_DarknetProcessor(char * label,int value)
{
 return 0;

}

unsigned char * getDataOutput_DarknetProcessor(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}

unsigned short * getDepth_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

unsigned char * getColor_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

int processData_DarknetProcessor()
{
 return 0;

}

int cleanup_DarknetProcessor()
{

 return 0;
}

int stop_DarknetProcessor()
{
 return 0;

}
