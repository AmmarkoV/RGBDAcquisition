#ifndef VIEWPOINT_CHANGE_H_INCLUDED
#define VIEWPOINT_CHANGE_H_INCLUDED



unsigned int FitImageInMask(unsigned char * imagePixels,unsigned char * maskPixels,unsigned int width , unsigned int height);

unsigned char * birdsEyeView(unsigned char * rgb,unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);
unsigned short * getVolumesBirdsEyeView(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

#endif // VIEWPOINT_CHANGE_H_INCLUDED
