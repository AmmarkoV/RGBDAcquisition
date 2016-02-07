#ifndef VIEWPOINT_CHANGE_H_INCLUDED
#define VIEWPOINT_CHANGE_H_INCLUDED


unsigned int FitImageInMask(struct Image * img, struct Image * mask);

unsigned char * birdsEyeView(unsigned char * rgb,unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);
unsigned short * getVolumesBirdsEyeView(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

#endif // VIEWPOINT_CHANGE_H_INCLUDED
