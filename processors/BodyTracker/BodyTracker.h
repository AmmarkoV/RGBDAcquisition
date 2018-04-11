#ifndef BodyDetector_H_INCLUDED
#define BodyDetector_H_INCLUDED


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



int initArgs_BodyDetector(int argc, char *argv[]);

int setConfigStr_BodyDetector(char * label,char * value);
int setConfigInt_BodyDetector(char * label,int value);


unsigned char * getDataOutput_BodyDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_BodyDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_BodyDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char  * getColor_BodyDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_BodyDetector();

int cleanup_BodyDetector();

int stop_BodyDetector();

#ifdef __cplusplus
}
#endif

#endif // BodyDetector_H_INCLUDED
