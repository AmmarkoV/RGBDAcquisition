#ifndef DISPARITYMAPPING_H_INCLUDED
#define DISPARITYMAPPING_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


int initArgs_FaceDetector(int argc, char *argv[]);

int setConfigStr_FaceDetector(char * label,char * value);
int setConfigInt_FaceDetector(char * label,int value);


unsigned char * getDataOutput_FaceDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_FaceDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_FaceDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * getColor_FaceDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_FaceDetector();

int cleanup_FaceDetector();

int stop_FaceDetector();

#ifdef __cplusplus
}
#endif


#endif // DISPARITYMAPPING_H_INCLUDED
