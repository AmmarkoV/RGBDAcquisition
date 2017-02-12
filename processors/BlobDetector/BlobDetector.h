#ifndef BLOBDETECTOR_H_INCLUDED
#define BLOBDETECTOR_H_INCLUDED


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


struct xyList * extractBlobsFromDepthMap(unsigned short * depth , unsigned int width , unsigned int height , unsigned int maxBlobs , unsigned int minBlobSize);
struct xyList * extractBlobsFromDepthMapNewBuffer(unsigned short * depth , unsigned int width , unsigned int height , unsigned int maxBlobs , unsigned int minBlobSize);


int initArgs_BlobDetector(int argc, char *argv[]);

int setConfigStr_BlobDetector(char * label,char * value);
int setConfigInt_BlobDetector(char * label,int value);


unsigned char * getDataOutput_BlobDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
int addDataInput_BlobDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel);


unsigned short * getDepth_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);
unsigned char * getColor_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel);


int processData_BlobDetector();

int cleanup_BlobDetector();

int stop_BlobDetector();

#ifdef __cplusplus
}
#endif

#endif // BLOBDETECTOR_H_INCLUDED
