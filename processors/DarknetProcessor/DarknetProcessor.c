#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DarknetProcessor.h"
#include "../../tools/ImageOperations/imageOps.h"
#include "../../3dparty/darknet/include/darknet.h"

unsigned int framesProcessed=0;


struct darknetContext
{
 image **alphabet;
 network * net;
 layer l;

 float nms;
 box *boxes;
 float **probs;

 float threshold;
};


struct darknetContext dc;

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};



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
                im.data[k*w*h + i*w + j] = (float) data[i*step + j*c + k]/255.;
            }
        }
    }

    //free(data);
    return im;
}




int init_yolo(
                const char *cfgfile,
                const char *weightfile,
                float thresh
               )
{
    if (!cfgfile)    { return 0; }
    if (!weightfile) { return 0; }

    dc.alphabet = load_alphabet();
    dc.net= parse_network_cfg(cfgfile);
    if(weightfile){
                   load_weights(dc.net, weightfile);
                  }
    dc.l = dc.net->layers[dc.net->n-1];

    set_batch_network(dc.net, 1);
    srand(2222222);


    dc.nms=.4;
    dc.boxes = calloc(dc.l.side * dc.l.side * dc.l.n, sizeof(box));
    dc.probs = calloc(dc.l.side * dc.l.side * dc.l.n, sizeof(float *));

    int j;
    for(j = 0; j < dc.l.side*dc.l.side*dc.l.n; ++j)
         { dc.probs[j] = calloc(dc.l.classes, sizeof(float *)); }


 return 1;
}

int initArgs_DarknetProcessor(int argc, char *argv[])
{
 char * cfgFile=0;
 char * weightFile=0;
 float threshold=0.2;


 unsigned int i=0;
 for (i=0; i<argc; i++)
 {
   if (strstr(argv[i],".cfg")!=0) { cfgFile=argv[i]; }
   if (strstr(argv[i],".weights")!=0) { weightFile=argv[i]; }
 }

 return init_yolo(
                   cfgFile,
                   weightFile,
                   threshold
                 );
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

int addDataInput_DarknetProcessor(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{

 fprintf(stderr,"addDataInput_DarknetProcessor %u (%ux%u) channels=%u\n" , stream , width, height,channels);
 if (stream==0)
 {
    fprintf(stderr,"casting image..\n");
    image im=load_image_from_buffer(data, width, height, channels);
    show_image(im, "original");
    //save_image(im, "original");
    fprintf(stderr,"resizing image..\n");
    image sized = resize_image(im, dc.net->w, dc.net->h);
    float *X = sized.data;

    fprintf(stderr,"detecting.. ");
    float *prediction = network_predict(dc.net, X);
    fprintf(stderr,"done\n");

    fprintf(stderr,"getting results ( %u outputs.. ) ..\n",dc.l.outputs);
    get_detection_boxes(dc.l, 1, 1, dc.threshold, dc.probs, dc.boxes, 0);

    if (dc.nms)
         { do_nms_sort(dc.boxes, dc.probs, dc.l.side* dc.l.side* dc.l.n, dc.l.classes, dc.nms); }

    printf("Objects:\n\n");
    draw_detections(im, dc.l.side * dc.l.side * dc.l.n  , dc.threshold , dc.boxes, dc.probs, 0, voc_names, dc.alphabet, 20/*classes*/);
    save_image(im, "predictions");
    show_image(im, "predictions");

    free_image(im);
    free_image(sized);

   ++framesProcessed;
  return 1;
 }
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

