#ifndef DISPARITYMAPPING_H_INCLUDED
#define DISPARITYMAPPING_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


int initArgs_DisparityMapping(int argc, char *argv[]);

int setConfigStr_DisparityMapping(char * label,char * value);
int setConfigInt_DisparityMapping(char * label,int value);


unsigned char * getDataOutput_DisparityMapping(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_DisparityMapping(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_DisparityMapping(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * getColor_DisparityMapping(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_DisparityMapping();

int cleanup_DisparityMapping();

int stop_DisparityMapping();

#ifdef __cplusplus
}
#endif


#endif // DISPARITYMAPPING_H_INCLUDED
