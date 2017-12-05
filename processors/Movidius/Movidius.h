#ifndef MOVIDIUS_H_INCLUDED
#define MOVIDIUS_H_INCLUDED


int initArgs_Movidius(int argc, char *argv[]);
int addDataInput_Movidius(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel) ;
int setConfigStr_Movidius(char * label,char * value) ;
int setConfigInt_Movidius(char * label,int value) ;
unsigned char * getDataOutput_Movidius(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel) ;
unsigned short * getDepth_Movidius(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel) ;

unsigned char * getColor_Movidius(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel) ;

int processData_Movidius() ;

int cleanup_Movidius() ;
int stop_Movidius();

#endif // MOVIDIUS_H_INCLUDED
