#ifndef PPM_H_INCLUDED
#define PPM_H_INCLUDED

int writePPM(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);
char * ReadPPM(char * filename,unsigned int *width,unsigned int *height);


#endif // PPM_H_INCLUDED
