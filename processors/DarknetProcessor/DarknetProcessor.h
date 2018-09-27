#ifndef DarknetProcessor_H_INCLUDED
#define DarknetProcessor_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

struct xyP { float x, y , z; } ;

struct xyList {
                 unsigned int maxListLength;
                 unsigned int listLength;
                 struct xyP* data;
                } ;


int initArgs_DarknetProcessor(int argc, char *argv[]);

int setConfigStr_DarknetProcessor(char * label,char * value);
int setConfigInt_DarknetProcessor(char * label,int value);


char * getDetectionLabel_DarknetProcessor(unsigned int detectionNumber);
float getDetectionProbability_DarknetProcessor(unsigned int detectionNumber);

unsigned char * getDataOutput_DarknetProcessor(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_DarknetProcessor(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * getColor_DarknetProcessor(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_DarknetProcessor();

int cleanup_DarknetProcessor();

int stop_DarknetProcessor();

#ifdef __cplusplus
}
#endif

#endif // DarknetProcessor_H_INCLUDED
