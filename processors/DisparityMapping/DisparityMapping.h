#ifndef DISPARITYMAPPING_H_INCLUDED
#define DISPARITYMAPPING_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif



unsigned char * getDataOutput(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput(unsigned int stream , unsigned char * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);
int processData();

#ifdef __cplusplus
}
#endif


#endif // DISPARITYMAPPING_H_INCLUDED
