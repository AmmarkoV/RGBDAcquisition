#ifndef OBSTACLEDETECTOR_H_INCLUDED
#define OBSTACLEDETECTOR_H_INCLUDED



int initArgs_ObstacleDetector(int argc, char *argv[]);

int setConfigStr_ObstacleDetector(char * label,char * value);
int setConfigInt_ObstacleDetector(char * label,int value);


unsigned char * getDataOutput_ObstacleDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_ObstacleDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * getColor_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_ObstacleDetector();

int cleanup_ObstacleDetector();

int stop_ObstacleDetector();

#endif // PERSONDETECTOR_H_INCLUDED
