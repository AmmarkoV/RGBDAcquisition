#ifndef PIXELFORMATCONVERSIONS_H_INCLUDED
#define PIXELFORMATCONVERSIONS_H_INCLUDED


int VideoFormatNeedsDecoding(int videoformat,int bitdepth);
int VideoFormatImplemented(int videoformat,int bitdepth);

void PrintOutPixelFormat(int pix_format);
void PrintOutCaptureMode(int cap_mode);
void PrintOutFieldType(int field_type);

int YUYV_2_RGB(unsigned char *yuv, unsigned char *rgb, unsigned int width, unsigned int height);

unsigned char *yuv420p_to_rgb24(int width, int height, unsigned char *pIn0, unsigned char *pOut0);
unsigned char *yuv420_to_rgb24(int width, int height, unsigned char *pIn0, unsigned char *pOut0);
unsigned char *yuv411p_to_rgb24(int width, int height, unsigned char *pIn0, unsigned char *pOut0);

int Convert2RGB24(unsigned char * ENC_frame , unsigned char * RGB_frame,unsigned int width,unsigned int height,int inp_videoformat,int inp_bitdepth);


#endif /* PIXELFORMATCONVERSIONS_H_INCLUDED*/
