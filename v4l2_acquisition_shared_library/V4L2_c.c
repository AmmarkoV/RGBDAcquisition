#include "V4L2_c.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>



static int xioctl(int fd,int request,void * arg)
{
   int r;
   do r = ioctl (fd, request, arg);
   while (-1 == r && EINTR == errno);
   return r;
};


int populate_v4l2intf(struct V4L2_c_interface * v4l2_interface,char * device,int method_used)
{
  if (v4l2_interface==0) { fprintf(stderr,"Populate populate_v4l2intf called with an unallocated v4l2intf \n"); return 0; }
  memset(v4l2_interface,0,sizeof(struct V4L2_c_interface));


  if (method_used==MMAP) { v4l2_interface->io=IO_METHOD_MMAP; } else
                         { v4l2_interface->io=IO_METHOD_MMAP; } /*If method used is incorrect just use MMAP :P*/

  v4l2_interface->fd=-1;

  strncpy(v4l2_interface->device,device,MAX_DEVICE_FILENAME);
  fprintf(stderr,"opening device: %s\n",device);

  struct stat st={0};
  if (-1 == stat (device, &st)) { fprintf (stderr, "Cannot identify '%s': %d, %s\n",device, errno, strerror (errno)); return 0;  }
  if (!S_ISCHR (st.st_mode))    { fprintf (stderr, "%s is no device\n", device); return 0; }

  //We just open the /dev/videoX file descriptor and start reading..!
  v4l2_interface->fd = open (v4l2_interface->device, O_RDWR /* required */ | O_NONBLOCK, 0);
  if (-1 == v4l2_interface->fd)
   {
        fprintf (stderr, "Cannot open '%s': %d, %s\n",device, errno, strerror (errno));
        return 0;
   }

   fprintf(stderr,"device opening ok \n");
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
                                                  if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,"Error VIDIOC_QBUF\n"); return 0; }
                                               }
                                          type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                          if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMON, &type)) { fprintf(stderr,"Error VIDIOC_STREAMON\n");  return 0; }
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
                                                      if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,"Error VIDIOC_QBUF\n"); return 0; }
                                                     }
                                               type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                                              if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMON, &type))  { fprintf(stderr,"Error VIDIOC_STREAMON\n"); return 0; }
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
                             if (-1 == xioctl (v4l2_interface->fd, VIDIOC_STREAMOFF, &type)) { fprintf(stderr,"Error VIDIOC_STREAMOFF"); return 0; }
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
                                                       default: fprintf(stderr,"Failed reading video stream\n");
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
                                                        default: fprintf(stderr,"Failed VIDIOC_DQBUF(1)\n");
                                                        return 0;
                                                      }
                                    }

                            //assert (buf.index < v4l2_interface->n_buffers);
                            if ( !(buf.index<v4l2_interface->n_buffers) ) { fprintf(stderr,"assert (buf.index < v4l2_interface->n_buffers); failed.."); return 0; }
                            if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,"Failed VIDIOC_QBUF\n"); return 0; }
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
                                                                                                          default: fprintf(stderr,"Failed VIDIOC_DQBUF(2)\n");
                                                                                                          return 0;
                                                                                                        }
                                                                                      }
                           for (i = 0; i < v4l2_interface->n_buffers; ++i)
                                             if (buf.m.userptr == (unsigned long) v4l2_interface->buffers[i].start && buf.length == v4l2_interface->buffers[i].length) break;
                           assert (i < v4l2_interface->n_buffers);
                           if (-1 == xioctl (v4l2_interface->fd, VIDIOC_QBUF, &buf)) { fprintf(stderr,"Failed VIDIOC_QBUF\n"); return 0; }
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


    if (0 == r) { fprintf (stderr, "Select call timed out\n"); return 0; }


    void *p=readFrame_v4l2intf(v4l2_interface);
    if (p!=0) return p;
    /* EAGAIN - continue select loop. */
  }
}



