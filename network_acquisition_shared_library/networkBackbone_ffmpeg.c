

int networkBackbone_startPushingToRemote(char * ip , int port , unsigned int width , unsigned int height)
{
  //ffserver
  //ffmpeg -i dota_stripped.avi -f matroska - | ffplay -
  //ffmpeg -i dota_stripped.avi -i http://localhost:8090/feed1.ffm
  //ffplay http://localhost:8090/feed1.ffm


   //ffmpeg -y -f rawvideo -vcodec rawvideo -s 420x360 -pix_fmt rgb24 -r 24 -i - -an -vcodec mpeg my_output_videofile.mp4

  //DUMP TO FILE DEBUG
  //output = popen ("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24 -i - -f mp4 -q:v 5 -an -vcodec mpeg4 test.mp4", "w");

  //output = popen ("ffmpeg -y -f lavfi -i aevalsrc=0  -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24 -i - -acodec copy -b:a 32k http://localhost:8090/feed1.ffm", "w");
  //-map 1:0
  //output = popen ("ffmpeg -y -f lavfi -i aevalsrc=0 -c:a pcm_s16le  -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24  -i - -vf \"format=yuv420p\" -g 60 -f flv http://localhost:8090/feed1.ffm", "w");
  output = popen ("ffmpeg -y -f rawvideo -vcodec rawvideo -s 640x480 -pix_fmt rgb24 -r 24  -i - -vf \"format=yuv420p\" -g 60 -f flv rtmp://a.rtmp.youtube.com/live2/", "w");
  if (!output)
    {
      fprintf (stderr,"incorrect parameters or too many files.\n");
      return 0;
    }
  return 1;
}

int networkBackbone_stopPushingToRemote(int frameServerID)
{
  if (pclose (output) != 0)
    {
      fprintf (stderr,"Could not run more or other error.\n");
      return 0;
    }
  return 1;
}




int networkBackbone_pushImageToRemote(int frameServerID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
  fprintf(stderr,"networkBackbone_pushImageToRemote\n");
  if (streamNumber==0) //Color
  {

      /*
      struct Image * img = createImageUsingExistingBuffer(width,height,channels,bitsperpixel,pixels);
      if (img!=0)
      {
       unsigned long compressedColorSize=64*1024; //64KBmax
       char * compressedPixels = (char* ) malloc(sizeof(char) * compressedColorSize);
       if (compressedPixels!=0)
        {
         WriteJPEGMemory(img,compressedPixels,&compressedColorSize);
         fprintf(stderr,"Compressed from %lu bytes\n",compressedColorSize);
         fwrite (compressedPixels, sizeof(char), compressedColorSize, output);
        }
      }
      */
     fwrite (pixels, sizeof(char), width*height*channels, output);

  }

  return 0;
}

