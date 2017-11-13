#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DarknetProcessor.h"
#include "../../tools/ImageOperations/imageOps.h"
#include "../../3dparty/darknet/include/darknet.h"

unsigned int framesProcessed=0;




void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    /*
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);*/
//    pthread_t detect_thread;
//    pthread_t fetch_thread;
}

int initArgs_DarknetProcessor(int argc, char *argv[])
{
 fprintf(stdout,"Frame,Blob,X,Y,Z,Samples\n");
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

int addDataInput_DarknetProcessor(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{

 fprintf(stderr,"addDataInput_DarknetProcessor %u (%ux%u)\n" , stream , width, height);
 if (stream==1)
 {

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

