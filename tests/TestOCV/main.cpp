#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

using namespace cv;


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


unsigned int simplePowCodecs(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}


int WritePPM(const char * filename,struct Image * pic)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);
    if (pic==0) { return 0; }
    if ( (pic->width==0) || (pic->height==0) || (pic->channels==0) || (pic->bitsperpixel==0) )
        {
          fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions ( %ux%u %u channels %u bpp\n",filename,pic->width , pic->height,pic->channels,pic->bitsperpixel);
          return 0;
        }
    if(pic->pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (pic->bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n"); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (fd!=0)
    {
        unsigned int n;
        if (pic->channels==3) fprintf(fd, "P6\n");
        else if (pic->channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",pic->channels);
            fclose(fd);
            return 1;
        }

        fprintf(fd, "%d %d\n%u\n", pic->width, pic->height , simplePowCodecs(2 ,pic->bitsperpixel)-1);

        float tmp_n = (float) pic->bitsperpixel/ 8;
        tmp_n = tmp_n *  pic->width * pic->height * pic->channels ;
        n = (unsigned int) tmp_n;

        fwrite(pic->pixels, 1 , n , fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }
    return 0;
}



void drawLine( cv::Mat img,
               unsigned int x1,unsigned int y1,
               unsigned int x2,unsigned int y2,
               unsigned int r,
               unsigned int g,
               unsigned int b
             )
{
  cv::Point start = cv::Point(x1,y1);
  cv::Point end = cv::Point(x2,y2);

  int thickness = 2;
  int lineType = 8;
  cv::line( img,
            start,
            end,
            cv::Scalar( b, g, r ),
            thickness,
            lineType );
}


void webcamProgram()
{


    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0)) return ;


    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("this is you, smile! :)", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();


}


int main (int argc,const char *argv[])
{
    cv::Mat image = cv::imread("lena.jpeg", CV_LOAD_IMAGE_COLOR);


    //OpenCV stuff BGR
    drawLine(image,30,30,40,40 , 255,0,0 );

    cv::imshow("Test",image);

    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);



    //Our stuff RGB
    struct Image ourImage;
    ourImage.bitsperpixel=8;
    ourImage.channels=3;
    ourImage.height=image.rows;
    ourImage.width=image.cols;
    ourImage.image_size = ourImage.height * ourImage.width * 3;
    ourImage.pixels = image.data;


    WritePPM("our.pnm",&ourImage);










    //webcamProgram();







    cv::waitKey(0);

    return 0;
}

