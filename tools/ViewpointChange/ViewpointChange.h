#ifndef VIEWPOINTCHANGE_H_INCLUDED
#define VIEWPOINTCHANGE_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif



unsigned int viewPointChange_fitImageInMask(unsigned char * img, unsigned char * mask , unsigned int width , unsigned int height);
unsigned char* viewPointChange_mallocTransformToBirdEyeView(unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height , unsigned int depthRange);
int viewPointChange_newFramesCompare(unsigned char *  prototype , unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height  , unsigned int depthRange );




#ifdef __cplusplus
}
#endif

#endif // VIEWPOINTCHANGE_H_INCLUDED
