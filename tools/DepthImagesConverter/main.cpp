#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

unsigned int simplePow(unsigned int base,unsigned int exp)
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


int saveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    //fprintf(stderr,"saveRawImageToFile(%s) called\n",filename);

    if ( (width==0) || (height==0) || (channels==0) || (bitsperpixel==0) ) { fprintf(stderr,"saveRawImageToFile(%s) called with zero dimensions\n",filename); return 0;}
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n"); return 0; }

    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            fclose(fd);
            return 1;
        }
        fprintf(fd, "%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
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


unsigned short * convert24bitTo16bit(void * input24bit,unsigned int width , unsigned int height)
{
  unsigned short * output = (unsigned short * )  malloc ( sizeof(unsigned short)  * width * height  );
  if (output == 0 ) { fprintf(stderr,"Could not allocate output for convert24bitTo16bit\n"); return 0; }

  unsigned char * traverser=(unsigned char * ) input24bit;
  unsigned short * casterUshort=0;
  unsigned int *   casterUint=0;
  unsigned short * outputPointer=output;

  unsigned char * endOfMem = traverser + width * height * 3;

  unsigned char * byte1;
    unsigned char * byte2;
      unsigned char * byte3;

  while ( traverser < endOfMem)
  {
    byte1 = traverser ++ ;
    byte2 = traverser ++ ;
    byte3 = traverser ++ ;

   // if ( (*byte1!=*byte2) || (*byte2!=*byte3) ) { fprintf(stderr,"!"); }

    casterUshort = (unsigned short*) byte1;
    casterUint = (unsigned int *) byte1;

    //unsigned int outBit = *casterUint;
    //*outputPointer = (unsigned short) outBit;
    *outputPointer = *casterUshort;

    ++outputPointer;
  }

 return output;
}





int main( int argc, char** argv )
{
    fprintf(stderr,"%u arguments\n",argc);
    if( argc < 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay ImageToStore" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

//    vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//    compression_params.push_back(9);

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.

    IplImage  *imageDepth = cvCreateImage( cvSize(image.cols,image.rows), IPL_DEPTH_16U , 1 );
    if (imageDepth==0) { fprintf(stderr,"Could not create a new Depth OpenCV Image\n");  return 0; }
    char *opencv_depth_pointer_retainer = imageDepth->imageData; // UGLY HACK
    imageDepth->imageData = (char *) image.data;


    IplImage *rdepth8  = cvCreateImage(cvSize(image.cols,image.rows), IPL_DEPTH_8U, 1);
    cvConvertScaleAbs(imageDepth, rdepth8, 255.0/2048,0);
    cvShowImage("RGBDAcquisition Depth", rdepth8);
    cvReleaseImage( &rdepth8 );

    //cvShowImage("RGBDAcquisition Depth RAW",imageDepth);
    imageDepth->imageData = opencv_depth_pointer_retainer; // UGLY HACK
    cvReleaseImage( &imageDepth );


    imshow( "Display window", image );                   // Show our image inside it.

    if( argc >= 3)
    {
     if (strstr(argv[2],".pnm")!=0)
         {
           fprintf(stderr,"Saving using my PNM writer , width %u , height %u , channel 1 , bitsperpixel %u , depth %u \n",image.cols,image.rows,8*image.step/image.cols,image.depth());


           //saveRawImageToFile(argv[2],(unsigned char*) image.data ,image.cols,image.rows,3, 8);

           unsigned short * temp  = convert24bitTo16bit(image.data,image.cols,image.rows);
           saveRawImageToFile(argv[2],(unsigned char*) temp ,image.cols,image.rows,1, 16 );
           free(temp);
         } else
         { imwrite(argv[2], image  ); }
    }
     else
     {
        waitKey(0);                                          // Wait for a keystroke in the window
     }

    return 0;
}
