#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "DarknetProcessor.h"
#include "recordOutput.h"



#define GPU 1
//if define GPU1 is not used then nothing will work as it is supposed to be if we are doing a GPU build..
#include "../../3dparty/darknet/include/darknet.h"

unsigned int framesProcessed=0;

struct darknetContext
{
 image **alphabet;
 network * net;

 float nms;
 box *boxes;
 float **probs;
 float **masks;
 char **names;

 float threshold;
 float hierarchyThreshold;
};


char * payload=0;
struct darknetContext dc={0};
FILE * fp=0;

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

    fprintf(stderr,"Allocating space ..\n");
    dc.boxes = (box *)   calloc( l.w * l.h * l.n, sizeof(box));
    dc.probs = (float **)calloc( l.w * l.h * l.n, sizeof(float *));

    int j;
    for(j = 0; j < l.w * l.h * l.n; ++j)
         { dc.probs[j] = (float *)calloc( l.classes+1, sizeof(float)); }

     if ( l.coords > 4)
        {
            dc.masks = calloc( l.w *  l.h * l.n, sizeof(float*));
            for(j = 0; j < l.w* l.h * l.n; ++j) dc.masks[j] = calloc( l.coords-4, sizeof(float *));
        }
    fprintf(stderr,"Done with initialization ..\n");
 return 1;
}

int initArgs_DarknetProcessor(int argc, char *argv[])
{
 char * cfgFile=0;
 char * weightFile=0;
 char * dataFile=0;
 float threshold=0.55; // was 0.35


 unsigned int i=0;
 for (i=0; i<argc; i++)
 {
   if (strstr(argv[i],".cfg")!=0)     { cfgFile=argv[i]; }
   if (strstr(argv[i],".weights")!=0) { weightFile=argv[i]; }
   if (strstr(argv[i],".data")!=0)    { dataFile=argv[i]; }
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

 fp=startLogging("surveilance.log");

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
    float *prediction = network_predict(dc.net /*Neural Net*/, sized.data /*Search Image*/);
    //fprintf(stderr,"done ( %u )\n",l.outputs);


    if (l.outputs!=0)
    {
     if(l.type == DETECTION)   {
                                //fprintf(stderr,"getting_detection_boxes .. ");
                                get_detection_boxes(l, 1, 1, dc.threshold, dc.probs, dc.boxes, 0);
                               }
                              else
     if (l.type == REGION)    {
                               //fprintf(stderr,"getting_region_boxes .. ");
                               get_region_boxes(l, im.w, im.h, dc.net->w, dc.net->h, dc.threshold, dc.probs, dc.boxes, 0, 0, 0, dc.hierarchyThreshold, 1);
                              } else
                              {
                               error("Last layer must produce detections\n");
                              }
    //fprintf(stderr,"done\n");

    if (dc.nms)
         { do_nms_sort(dc.boxes, dc.probs, l.w*l.h*l.n, l.classes, dc.nms); }

    //printf("Objects (%u classes):\n\n",l.classes);

    unsigned int detections =  l.w * l.h * l.n;


   time_t clock = time(NULL);
   struct tm * ptm = gmtime ( &clock );

    draw_detections(im, detections , dc.threshold, dc.boxes, dc.probs, dc.masks, dc.names , dc.alphabet, l.classes);

    unsigned int i=0,j=0;
    for(i = 0; i <detections; ++i)
    {
        for(j = 0; j < l.classes; ++j){
            if (dc.probs[i][j] >  dc.threshold)
            {
               logEvent(
                        fp,
                        ptm,
                        framesProcessed,
                        dc.boxes[i].x,
                        dc.boxes[i].y,
                        dc.boxes[i].w,
                        dc.boxes[i].h,
                        dc.names[j],
                        dc.probs[i][j]*100
                      );


               if (strcmp(dc.names[j],"person")==0)
                  {
                    if (payload!=0)
                    {
                      int i=system(payload);
                      if (i!=0)
                      { fprintf(stderr,"Payload (%s) failed..\n",payload); }
                     }
                  }
            }
        }
    } // End for loop

    fflush(fp);


    char recordFile[512]={0};
    snprintf(recordFile,512,"record_%u",framesProcessed);
    save_image(im,recordFile);



    //show_image(im, "predictions");
    } else
    {
     fprintf(stderr,"Failed to run network ( prediction points to %p )..\n",prediction);
    }

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
 stopLogging(fp);
 return 0;

}

