#include "V4L2_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

static int xioctl(int fd,int request,void * arg)
{
   int r;
   do r = ioctl (fd, request, arg);
   while (-1 == r && EINTR == errno);
   return r;
};

int populateAndStart_v4l2intf(struct V4L2_c_interface * v4l2_interface,const char * device,int method_used)
{
  if (v4l2_interface==0) { fprintf(stderr,"Populate populate_v4l2intf called with an unallocated v4l2intf \n"); return 0; }
  memset(v4l2_interface,0,sizeof(struct V4L2_c_interface));


  if (method_used==MMAP) { v4l2_interface->io=IO_METHOD_MMAP; } else
                         { v4l2_interface->io=IO_METHOD_MMAP; } /*If method used is incorrect just use MMAP :P*/

  v4l2_interface->fd=-1;

  strncpy(v4l2_interface->device,device,MAX_DEVICE_FILENAME);
  fprintf(stderr,"Opening device: %s\n",device);

  struct stat st={0};
  if (-1 == stat (device, &st)) { fprintf (stderr, "Cannot identify '%s': %d, %s\n",device, errno, strerror (errno)); return 0;  }
  if (!S_ISCHR (st.st_mode))    { fprintf (stderr, "%s is no device\n", device); return 0; }

  //We just open the /dev/videoX file descriptor and start reading..!
  v4l2_interface->fd = open (v4l2_interface->device, O_RDWR /* required */ /*| O_NONBLOCK non block*/, 0);
  if (-1 == v4l2_interface->fd)
   {
        fprintf (stderr, RED "Cannot open '%s': %d, %s\n" NORMAL ,device, errno, strerror (errno));
        return 0;
   }

   fprintf(stderr,GREEN "Device opening ok \n" NORMAL);
   return 1;
}

int destroy_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  if (-1 == close (v4l2_interface->fd)) { fprintf(stderr,"Could not close v4l2_interface file descriptor\n"); return 0; }
  v4l2_interface->fd = -1;
  return 1;
}


int getFileDescriptor_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  return v4l2_interface->fd;
}

int getcap_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_capability *cap)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QUERYCAP,cap)) { return 0; }  else { return 1; }
}

/* Set frame format. see: http://staff.science.uva.nl/~bterwijn/Projects/V4L2/v4l2_website/v4l2spec.bytesex.org/spec-single/v4l2.html#V4L2-FORMAT */
int setfmt_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_format fmt)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_S_FMT, &fmt)) { return 0; } else { return 1; }
}

/* Get frame format. see: http://staff.science.uva.nl/~bterwijn/Projects/V4L2/v4l2_website/v4l2spec.bytesex.org/spec-single/v4l2.html#V4L2-FORMAT */
int getfmt_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_format *fmt)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_G_FMT, fmt)) { return 0; }  else { return 1; }
}

/* Query control information on brightness, contrast, saturation, etc. see: http://staff.science.uva.nl/~bterwijn/Projects/V4L2/v4l2_website/v4l2spec.bytesex.org/spec-single/v4l2.html#V4L2-QUERYCTRL */
int queryctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_queryctrl *ctrl)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QUERYCTRL, ctrl)) { return 0; } else { return 1; }
}

/* Set control information on brightness, contrast, saturation, etc. * see: http://staff.science.uva.nl/~bterwijn/Projects/V4L2/v4l2_website/v4l2spec.bytesex.org/spec-single/v4l2.html#V4L2-VIDIOC_S_CTRL */
int setctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_control  control)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_S_CTRL, &control)) { return 0; } else { return 1; }
}

/* Get control information on brightness, contrast, saturation, etc. see: http://staff.science.uva.nl/~bterwijn/Projects/V4L2/v4l2_website/v4l2spec.bytesex.org/spec-single/v4l2.html#V4L2-VIDIOC_G_CTRL */
int getctrl_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_control *control)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_G_CTRL, &control)) { return 0; } else { return 1; }
}


int setsparam_v4l2intf(struct V4L2_c_interface * v4l2_interface,struct v4l2_streamparm *param)
{
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_S_PARM, param)) { return 0; } else { return 1; }
}


int setFramerate_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int fps)
{
  #warning "setFramerate is not working "
  struct v4l2_streamparm parm={0};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = (unsigned int) fps;

  return setsparam_v4l2intf(v4l2_interface,&parm);
}


/*
      THESE ARE THE GETTERS AND SETTERS , WE NOW MOVE TO BASIC FUNCTIONALITY
*/

int initread_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int buffer_size)
{
  v4l2_interface->buffers = (struct buffer *)(calloc (1, sizeof (*v4l2_interface->buffers)));
  if (!v4l2_interface->buffers) { fprintf (stderr, "Out of memory , while initializing for read operations\n"); return 0; }

  v4l2_interface->buffers[0].length = buffer_size;
  v4l2_interface->buffers[0].start = malloc (buffer_size);
  if (!v4l2_interface->buffers[0].start)  { fprintf (stderr, "Out of memory\n"); return 0;  }
  return 1;
}

int inituserp_v4l2intf(struct V4L2_c_interface * v4l2_interface,unsigned int buffer_size)
{
  struct v4l2_requestbuffers req;
  CLEAR (req);
  req.count = 4; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_USERPTR;
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_REQBUFS, &req))
  {
    if (EINVAL == errno) { fprintf (stderr, "%s does not support user pointer i/o\n", v4l2_interface->device); return 0;  } else
                         { fprintf(stderr,"Error while calling VIDIOC_REQBUFS\n"); return 0; }
  }

  v4l2_interface->buffers = (struct buffer *)(calloc (4, sizeof (*v4l2_interface->buffers)));
  if (!v4l2_interface->buffers) { fprintf (stderr, "Could not allocate memory for a user video buffer \n"); return 0; }

  for (v4l2_interface->n_buffers = 0; v4l2_interface->n_buffers < 4; ++v4l2_interface->n_buffers)
  {
    v4l2_interface->buffers[v4l2_interface->n_buffers].length = buffer_size;
    v4l2_interface->buffers[v4l2_interface->n_buffers].start = malloc (buffer_size);

    if (!v4l2_interface->buffers[v4l2_interface->n_buffers].start) { fprintf (stderr, "Could not allocate memory for a user video buffer \n");  return 0; }
  }
  return 1;
}



int initmmap_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  struct v4l2_requestbuffers req;
  CLEAR (req);
  req.count = 4; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_MMAP;
  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_REQBUFS, &req))
  {
    if (EINVAL == errno) { fprintf (stderr, "%s does not support memory mapping\n", v4l2_interface->device); return 0; } else
                         { fprintf(stderr,"Error while calling VIDIOC_REQBUFS\n"); return 0; }
  }

  if (req.count < 2)    { fprintf (stderr, "Insufficient buffer memory on %s\n",v4l2_interface->device); return 0;  }

  v4l2_interface->buffers = (struct buffer *)(calloc (req.count, sizeof (*v4l2_interface->buffers)));
  if (!v4l2_interface->buffers) { fprintf (stderr, "Out of memory\n"); return 0;  }

  for (v4l2_interface->n_buffers = 0; v4l2_interface->n_buffers < req.count; ++v4l2_interface->n_buffers)
  {
    struct v4l2_buffer buf;
    CLEAR (buf);
    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = v4l2_interface->n_buffers;
    if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QUERYBUF, &buf)) { fprintf(stderr,"Error while calling VIDIOC_QUERYBUF\n"); return 0; }

    v4l2_interface->buffers[v4l2_interface->n_buffers].length = buf.length;
    v4l2_interface->buffers[v4l2_interface->n_buffers].start =
      mmap (NULL /* start anywhere */,
	        buf.length,
	        PROT_READ | PROT_WRITE /* required */,
	        MAP_SHARED /* recommended */,
	        v4l2_interface->fd,
	        buf.m.offset);

    if (MAP_FAILED == v4l2_interface->buffers[v4l2_interface->n_buffers].start) { fprintf(stderr,"Error mapping memory for video\n"); return 0; }
  }
  return 1;
}



int initBuffers_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  struct v4l2_format fmt;
  CLEAR (fmt);
  fmt.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  getfmt_v4l2intf(v4l2_interface,&fmt);

  switch (v4l2_interface->io)
  {                                                                      /*Breaks not needed they are there as a reminder :P*/
    case IO_METHOD_READ:    return initread_v4l2intf(v4l2_interface,fmt.fmt.pix.sizeimage);   break;
    case IO_METHOD_MMAP:    return initmmap_v4l2intf(v4l2_interface);                         break;
    case IO_METHOD_USERPTR: return inituserp_v4l2intf(v4l2_interface,fmt.fmt.pix.sizeimage);  break;
  }

  return 0;
}


int freeBuffers_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  int i = 0;
  switch (v4l2_interface->io)
  {
    case IO_METHOD_READ:    free(v4l2_interface->buffers[0].start); break;
    case IO_METHOD_MMAP:    for (i = 0; i < (int)v4l2_interface->n_buffers; ++i) { if (-1 == munmap ( v4l2_interface->buffers[i].start, v4l2_interface->buffers[i].length)) { fprintf(stderr,"Error freeing buffers \n"); return 0; } } break;
    case IO_METHOD_USERPTR: for (i = 0; i < (int)v4l2_interface->n_buffers; ++i) { free (v4l2_interface->buffers[i].start); } break;
  }
  free (v4l2_interface->buffers);

 return 1;
}







int startCapture_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  enum v4l2_buf_type type;
  unsigned int i;

  switch (v4l2_interface->io)
  {
                    case IO_METHOD_READ:  /* Nothing to do. */ break;
                    case IO_METHOD_MMAP:
                                          for (i = 0; i < v4l2_interface->n_buffers; ++i)
                                               {
                                                  struct v4l2_buffer buf;
                                                  CLEAR (buf);
                                                  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                                  buf.memory = V4L2_MEMORY_MMAP;
                                                  buf.index = i;
                                                  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,RED "Error VIDIOC_QBUF\n" NORMAL); return 0; }
                                               }
                                          type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                          if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMON, &type)) { fprintf(stderr,RED "Error VIDIOC_STREAMON\n" NORMAL);  return 0; }
                     break;

                     case IO_METHOD_USERPTR:
                                              for (i = 0; i < v4l2_interface->n_buffers; ++i)
                                                    {
                                                      struct v4l2_buffer buf;
                                                      CLEAR (buf);
                                                      buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                                      buf.memory      = V4L2_MEMORY_USERPTR;
                                                      buf.index       = i;
                                                      buf.m.userptr   = (unsigned long) v4l2_interface->buffers[i].start;
                                                      buf.length      = v4l2_interface->buffers[i].length;
                                                      if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,RED "Error VIDIOC_QBUF\n" NORMAL); return 0; }
                                                     }
                                               type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                              if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMON, &type))  { fprintf(stderr,RED "Error VIDIOC_STREAMON\n" NORMAL); return 0; }
                    break;
  }

  return 1;
}


int stopCapture_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  enum v4l2_buf_type type;
  switch (v4l2_interface->io)
  {
  case IO_METHOD_READ: /* Nothing to do. */ break;
  case IO_METHOD_MMAP:
  case IO_METHOD_USERPTR:   /*Common for MMAP and userptr*/
                             type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                             if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMOFF, &type)) { fprintf(stderr,RED "Error VIDIOC_STREAMOFF" NORMAL); return 0; }
                             break;
  }
  return 1;
}




void * readFrame_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  struct v4l2_buffer buf;
  unsigned int i;

  switch (v4l2_interface->io)
  {

    case IO_METHOD_READ:
                            if (-1 == read (v4l2_interface->fd, v4l2_interface->buffers[0].start, v4l2_interface->buffers[0].length))
                                  {
                                     switch (errno) {
                                                       case EAGAIN: return 0;
                                                       case EIO: /* Could ignore EIO, see spec. */ /* fall through */
                                                       default: fprintf(stderr,RED "Failed reading video stream\n"NORMAL );
                                                       return 0;
                                                     }
                                  }
     //Successful IO_METHOD_READ..
     return v4l2_interface->buffers[0].start;
    break;

    case IO_METHOD_MMAP:
                            CLEAR (buf);
                            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                            buf.memory = V4L2_MEMORY_MMAP;
                            if (-1 == xioctl (v4l2_interface->fd, VIDIOC_DQBUF, &buf))
                                   {
                                     switch (errno) {
                                                        case EAGAIN: return 0;
                                                        case EIO: /* Could ignore EIO, see spec. */ /* fall through */
                                                        default: fprintf(stderr,RED "Failed VIDIOC_DQBUF(1)\n"NORMAL);
                                                        return 0;
                                                      }
                                    }

                            //assert (buf.index < v4l2_interface->n_buffers);
                            if ( !(buf.index<v4l2_interface->n_buffers) ) { fprintf(stderr,RED "assert (buf.index < v4l2_interface->n_buffers); failed.." NORMAL); return 0; }
                            if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,RED "Failed VIDIOC_QBUF\n" NORMAL); return 0; }
     //Successful IO_METHOD_MMAP..
     return v4l2_interface->buffers[buf.index].start;
    break;

  case IO_METHOD_USERPTR:
                           CLEAR (buf);
                           buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                           buf.memory = V4L2_MEMORY_USERPTR;
                           if (-1 == xioctl (v4l2_interface->fd, VIDIOC_DQBUF, &buf)) {
                                                                                        switch (errno) {
                                                                                                          case EAGAIN: return 0;
                                                                                                          case EIO: /* Could ignore EIO, see spec. */ /* fall through */
                                                                                                          default: fprintf(stderr,RED "Failed VIDIOC_DQBUF(2)\n" NORMAL);
                                                                                                          return 0;
                                                                                                        }
                                                                                      }
                           for (i = 0; i < v4l2_interface->n_buffers; ++i)
                                             if (buf.m.userptr == (unsigned long) v4l2_interface->buffers[i].start && buf.length == v4l2_interface->buffers[i].length) break;
                           assert (i < v4l2_interface->n_buffers);
                           if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,RED "Failed VIDIOC_QBUF\n" NORMAL); return 0; }
    //Successful IO_METHOD_USERPTR..
    return (void *)(buf.m.userptr);
    break;

  }

  return 0;
}







void * getFrame_v4l2intf(struct V4L2_c_interface * v4l2_interface)
{
  for (;;)
  {
    fd_set fds;
    struct timeval tv;
    int r;
    FD_ZERO (&fds);
    FD_SET (v4l2_interface->fd, &fds);

    /* Timeout for frame grabbing .. */
    tv.tv_sec = TIMEOUT_SEC;
    tv.tv_usec = TIMEOUT_USEC;
    /* ----------------------------- */

    r = select (v4l2_interface->fd + 1, &fds, 0, 0, &tv);
    if (-1 == r) { if (EINTR == errno) continue; fprintf(stderr,"Error selecting stream\n");  return 0; }


    if (0 == r) { fprintf (stderr, YELLOW "Select call timed out\n" NORMAL); return 0; }


    void *p=readFrame_v4l2intf(v4l2_interface);
    if (p!=0) return p;
    /* EAGAIN - continue select loop. */
  }
}


/*
    ----------------------------------------------------------
    ----------------------------------------------------------

                       Printing Data from V4L2

    ----------------------------------------------------------
    ----------------------------------------------------------
*/



static char * num2s(unsigned num,char * buf)
{
	//char buf[10];
	sprintf(buf, "%08x", num);
	return buf;
}

char * buftype2s(int type)
{
	switch (type) {
	case 0:
		return "Invalid";
	case V4L2_BUF_TYPE_VIDEO_CAPTURE:
		return "Video Capture";
	case V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE:
		return "Video Capture Multiplanar";
	case V4L2_BUF_TYPE_VIDEO_OUTPUT:
		return "Video Output";
	case V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE:
		return "Video Output Multiplanar";
	case V4L2_BUF_TYPE_VIDEO_OVERLAY:
		return "Video Overlay";
	case V4L2_BUF_TYPE_VBI_CAPTURE:
		return "VBI Capture";
	case V4L2_BUF_TYPE_VBI_OUTPUT:
		return "VBI Output";
	case V4L2_BUF_TYPE_SLICED_VBI_CAPTURE:
		return "Sliced VBI Capture";
	case V4L2_BUF_TYPE_SLICED_VBI_OUTPUT:
		return "Sliced VBI Output";
	case V4L2_BUF_TYPE_VIDEO_OUTPUT_OVERLAY:
		return "Video Output Overlay";
	default:
		return "Unknown ( Error ) ";// + num2s(type) + ")";
	}
}

char * fcc2s(unsigned int val,char * s)
{
    //char s[10];
	s[0]= val & 0xff;
	s[1]= (val >> 8) & 0xff;
	s[2]= (val >> 16) & 0xff;
	s[3]= (val >> 24) & 0xff;
	return s;
}

char * field2s(int val)
{
	switch (val) {
	case V4L2_FIELD_ANY:
		return "Any";
	case V4L2_FIELD_NONE:
		return "None";
	case V4L2_FIELD_TOP:
		return "Top";
	case V4L2_FIELD_BOTTOM:
		return "Bottom";
	case V4L2_FIELD_INTERLACED:
		return "Interlaced";
	case V4L2_FIELD_SEQ_TB:
		return "Sequential Top-Bottom";
	case V4L2_FIELD_SEQ_BT:
		return "Sequential Bottom-Top";
	case V4L2_FIELD_ALTERNATE:
		return "Alternating";
	case V4L2_FIELD_INTERLACED_TB:
		return "Interlaced Top-Bottom";
	case V4L2_FIELD_INTERLACED_BT:
		return "Interlaced Bottom-Top";
	default:
		return "Unknown ( Error ) ";// + num2s(val) + ")";
	}
}

char * colorspace2s(int val)
{
	switch (val) {
	case V4L2_COLORSPACE_SMPTE170M:
		return "Broadcast NTSC/PAL (SMPTE170M/ITU601)";
	case V4L2_COLORSPACE_SMPTE240M:
		return "1125-Line (US) HDTV (SMPTE240M)";
	case V4L2_COLORSPACE_REC709:
		return "HDTV and modern devices (ITU709)";
	case V4L2_COLORSPACE_BT878:
		return "Broken Bt878";
	case V4L2_COLORSPACE_470_SYSTEM_M:
		return "NTSC/M (ITU470/ITU601)";
	case V4L2_COLORSPACE_470_SYSTEM_BG:
		return "PAL/SECAM BG (ITU470/ITU601)";
	case V4L2_COLORSPACE_JPEG:
		return "JPEG (JFIF/ITU601)";
	case V4L2_COLORSPACE_SRGB:
		return "SRGB";
	default:
		return "Unknown ( Error ) ";// + num2s(val) + ")";
	}
}



char * frmtype2s(unsigned type)
{
	static const char *types[] = { "Unknown", "Discrete", "Continuous", "Stepwise" };
	if (type > 3) type = 0;
	return types[type];
}

char * fract2sec(const struct v4l2_fract *f,char * buf)
{
	//char buf[100];
	sprintf(buf, "%.3f", (1.0 * f->numerator) / f->denominator);
	return buf;
}

char * fract2fps(const struct v4l2_fract *f,char * buf)
{
	//char buf[100];
	sprintf(buf, "%.3f", (1.0 * f->denominator) / f->numerator);
	return buf;
}


void print_frmsize(const struct v4l2_frmsizeenum * frmsize, const char *prefix)
{
	printf("%s\tSize: %s ", prefix, frmtype2s(frmsize->type));
	if (frmsize->type == V4L2_FRMSIZE_TYPE_DISCRETE) {
		printf("%dx%d", frmsize->discrete.width, frmsize->discrete.height);
	} else
	if (frmsize->type == V4L2_FRMSIZE_TYPE_STEPWISE)
	  {
		printf("%dx%d - %dx%d with step %d/%d",
				frmsize->stepwise.min_width,
				frmsize->stepwise.min_height,
				frmsize->stepwise.max_width,
				frmsize->stepwise.max_height,
				frmsize->stepwise.step_width,
				frmsize->stepwise.step_height);
	  }
	printf("\n");
}


void print_frmival(const struct v4l2_frmivalenum * frmival, const char *prefix)
{
    char buf1[100];
    char buf2[100];
	//printf("%s\tInterval: %s ", prefix, frmtype2s(frmival->type).c_str());
	if (frmival->type == V4L2_FRMIVAL_TYPE_DISCRETE)
    {
       //printf("%ss (%s fps)\n", fract2sec(frmival,buf1), fract2fps(frmival,buf2) );
	} else
	if (frmival->type == V4L2_FRMIVAL_TYPE_STEPWISE)
    {/*
		printf("%ss - %ss with step %ss (%s-%s fps)\n",
				fract2sec(frmival->stepwise.min,buf),
				fract2sec(frmival->stepwise.max,buf),
				fract2sec(frmival->stepwise.step,buf),
				fract2fps(frmival->stepwise.max,buf),
				fract2fps(frmival->stepwise.min,buf)   );*/
	}
}

// enum v4l2_buf_type
void print_video_formats_ext(int fd,int type)
{
	struct v4l2_fmtdesc fmt;
	struct v4l2_frmsizeenum frmsize;
	struct v4l2_frmivalenum frmival;


    char ps[100];
	fmt.index = 0;
	fmt.type = type;
	fprintf(stderr,"Printing Video Formats \n");
	while (ioctl(fd, VIDIOC_ENUM_FMT, &fmt) >= 0)
    {
		printf("\tIndex       : %d\n", fmt.index);
		printf("\tType        : %s\n", buftype2s(type));
		printf("\tPixel Format: '%s'", fcc2s(fmt.pixelformat,ps));
		if (fmt.flags) { printf(" (%s)", fmtdesc2s(fmt.flags)); }
		printf("\n");
		printf("\tName        : %s\n", fmt.description);
		frmsize.pixel_format = fmt.pixelformat;
		frmsize.index = 0;
		while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) >= 0)
		{
			print_frmsize(&frmsize, "\t");
			if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE)
			 {
				frmival.index = 0;
				frmival.pixel_format = fmt.pixelformat;
				frmival.width = frmsize.discrete.width;
				frmival.height = frmsize.discrete.height;
				while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) >= 0)
				{
					print_frmival(&frmival, "\t\t");
					frmival.index++;
				}
		     }
			frmsize.index++;
	     }
		printf("\n");
		fmt.index++;
	}
}


