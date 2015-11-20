#ifndef DISPARITYMAPPING_H_INCLUDED
#define DISPARITYMAPPING_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


int setConfigStr_DisparityMapping(char * label,char * value);
int setConfigInt_DisparityMapping(char * label,int value);


unsigned char * getDataOutput_DisparityMapping(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_DisparityMapping(unsigned int stream , unsigned char * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * processData_GetDepth(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * processData_GetColor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_DisparityMapping();

#ifdef __cplusplus
}
#endif


#endif // DISPARITYMAPPING_H_INCLUDED
