#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

typedef unsigned char BYTE;


struct Image
{
  unsigned char * pixels;
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  unsigned int image_size;
  unsigned long timestamp;
};

 


#ifdef __cplusplus
}
#endif

#endif // IMAGE_H_INCLUDED

