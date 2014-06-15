#ifndef VIEWPOINTCHANGE_H_INCLUDED
#define VIEWPOINTCHANGE_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


unsigned char * viewPointChange_ReadPPM(char * filename,unsigned int * width , unsigned int * height , unsigned int * channels,unsigned int * bitsperpixel,char read_only_header);

unsigned int viewPointChange_fitImageInMask(unsigned char * img, unsigned char * mask , unsigned int width , unsigned int height);
unsigned char* viewPointChange_mallocTransformToBirdEyeView(unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height , unsigned int depthRange);
int viewPointChange_newFramesCompare(unsigned char *  prototype , unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height  , unsigned int depthRange );

unsigned int viewPointChange_countDepths(unsigned short *  depth, unsigned int imageWidth , unsigned int imageHeight  ,
                                unsigned int x , unsigned int y , unsigned int width , unsigned int height ,
                                unsigned int depthRange );


#ifdef __cplusplus
}
#endif

#endif // VIEWPOINTCHANGE_H_INCLUDED
