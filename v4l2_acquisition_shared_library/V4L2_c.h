#ifndef V4L2_C_H_INCLUDED
#define V4L2_C_H_INCLUDED

#include <linux/types.h>
#include <linux/videodev2.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <errno.h>

#include <asm/types.h>          /* for videodev2.h */



#define CLEAR(x) memset (&(x), 0, sizeof (x))

#define MAX_DEVICE_FILENAME 512

#define TIMEOUT_SEC 10
#define TIMEOUT_USEC 0


enum v4l2_method_used
{
    READ ,
    MMAP ,
    USERPTR
};

typedef enum
{
  IO_METHOD_READ,
  IO_METHOD_MMAP,
  IO_METHOD_USERPTR,
} io_method;


struct buffer
{
  void * start;
  size_t length;
};

struct V4L2_c_interface
{
  char device[MAX_DEVICE_FILENAME];
  io_method io;
  int fd;
  struct buffer *buffers;
  unsigned int n_buffers;
};

int populateAndStart_v4l2intf(struct V4L2_c_interface * v4l2_interface,char * device,int method_used);
int destroy_v4l2intf(struct V4L2_c_interface * v4l2_interface);
int getFileDescriptor_v4l2intf(struct V4L2_c_interface * v4l2_interface);

int getcap_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_capability *cap);

int setfmt_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_format fmt) ;
int getfmt_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_format *fmt) ;

int queryctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_queryctrl *ctrl);
int setctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_control  control);
int getctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_control *control);

int setsparam_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_streamparm *param);
int setFramerate_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int fps);


int initread_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int buffer_size);
int inituserp_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int buffer_size);
int initmmap_v4l2intf(struct V4L2_c_interface * v4l2_interface);

int initBuffers_v4l2intf(struct V4L2_c_interface * v4l2_interface);
int freeBuffers_v4l2intf(struct V4L2_c_interface * v4l2_interface);

int startCapture_v4l2intf(struct V4L2_c_interface * v4l2_interface);
int stopCapture_v4l2intf(struct V4L2_c_interface * v4l2_interface);

void print_video_formats_ext(int fd, enum v4l2_buf_type type);

void * getFrame_v4l2intf(struct V4L2_c_interface * v4l2_interface);


#endif // V4L2_C_H_INCLUDED
