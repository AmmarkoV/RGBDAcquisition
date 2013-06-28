#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 3)
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
    imshow( "Display window", image );                   // Show our image inside it.
    imwrite(argv[2], image  );

    //waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
