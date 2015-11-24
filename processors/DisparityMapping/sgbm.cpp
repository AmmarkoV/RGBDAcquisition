#include "sgbm.h"


#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "stereo_calibrate.h"

int flipLeftRight=1;
unsigned int shiftYLeft=0;
unsigned int shiftYRight=17;

using namespace cv;

int doSGBM(unsigned char * colorFrame , unsigned int colorWidth ,unsigned int colorHeight )
{
    cv::Mat rgbImg(colorHeight,colorWidth,CV_8UC3,colorFrame);
    //cv::Mat depthImg(depthHeight,depthWidth,CV_16UC1,depthFrame);

    std::cerr<<"Doing Left split of images..\n";

    cv::Mat tmpleftImage;
    cv::Mat tmprightImage;
    cv::Rect leftROI;
    cv::Rect rightROI;


    leftROI = cv::Rect(0,0,(rgbImg.cols/2)-1,rgbImg.rows-1);
    rightROI = cv::Rect(rgbImg.cols/2,0,(rgbImg.cols/2)-1,rgbImg.rows-1);

    if (flipLeftRight)
    {
     tmpleftImage= cv::Mat(rgbImg,rightROI);
     tmprightImage= cv::Mat(rgbImg,leftROI);
    }else
    {
     tmpleftImage= cv::Mat(rgbImg,leftROI);
     tmprightImage= cv::Mat(rgbImg,rightROI);
    }

    cv::Mat bgrleftImage = cv::Mat::zeros(tmpleftImage.size(), tmpleftImage.type());
    if (shiftYLeft==0) { bgrleftImage=tmpleftImage;  } else
                       {
                         tmpleftImage(cv::Rect(0,shiftYLeft, tmpleftImage.cols,tmpleftImage.rows-shiftYLeft)).copyTo(bgrleftImage(cv::Rect(0,0,tmpleftImage.cols,tmpleftImage.rows-shiftYLeft)));
                       }

    cv::Mat bgrrightImage = cv::Mat::zeros(tmprightImage.size(), tmprightImage.type());
    if (shiftYRight==0) { bgrrightImage=tmprightImage;  } else
                       {
                        tmprightImage(cv::Rect(0,shiftYRight, tmprightImage.cols,tmprightImage.rows-shiftYRight)).copyTo(bgrrightImage (cv::Rect(0,0,tmprightImage.cols,tmprightImage.rows-shiftYRight)));
                       }

    cv::Mat leftImage;
    cv::Mat rightImage;

     cv::cvtColor(bgrleftImage,leftImage, cv::COLOR_RGB2BGR);
     cv::cvtColor(bgrrightImage,rightImage, cv::COLOR_RGB2BGR);



    //std::cerr<<"Doing drawing..\n";
    //cv::imshow("testLeft",leftImage);
    //cv::imshow("testRight",rightImage);

    double alpha = 0.5;
    double beta = ( 1.0 - alpha );
    cv::Mat blend ;
    cv::addWeighted( leftImage, alpha, rightImage , beta, 0.0, blend);
    cv::imshow("blending",blend);


    //cv::waitKey(5);

    std::cerr<<"Done with preliminary stuff..\n";

    int argc=0;
    char * argv=0;

    //return 1;

    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display=";
    const char* scale_opt = "--scale=";


    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* intrinsic_filename = 0;
    const char* extrinsic_filename = 0;
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 45, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    StereoBM bm;
    StereoSGBM sgbm;
    StereoVar var;


    if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename == 0 && point_cloud_filename )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 ,img2 ;
    img1 = leftImage; //imread(img1_filename, color_mode);
    img2 = rightImage; //imread(img2_filename, color_mode);

    if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( intrinsic_filename )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename);
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        fs.open(extrinsic_filename, CV_STORAGE_READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename);
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = 15;
    bm.state->speckleWindowSize = 100;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;
    sgbm.speckleRange = bm.state->speckleRange;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;

    var.levels = 3;									// ignored with USE_AUTO_PARAMS
	var.pyrScale = 0.5;								// ignored with USE_AUTO_PARAMS
	var.nIt = 25;
	var.minDisp = -numberOfDisparities;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV;	// ignored with USE_AUTO_PARAMS
	var.cycle = var.CYCLE_V;						// ignored with USE_AUTO_PARAMS
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm(img1, img2, disp);
    else if( alg == STEREO_VAR ) {
        var(img1, img2, disp);
	}
    else if( alg == STEREO_SGBM || alg == STEREO_HH )
        sgbm(img1, img2, disp);
    t = getTickCount() - t;
    printf("OpenCV Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);
    if( !no_display )
    {
        //namedWindow("left", 1);
        //imshow("left", img1);

        //namedWindow("right", 1);
        //imshow("right", img2);

        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        //printf("press any key to continue...");
        //fflush(stdout);
       // waitKey();
        //printf("\n");
    }

    if(disparity_filename)
        imwrite(disparity_filename, disp8);

    if(point_cloud_filename)
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        printf("\n");
    }



}
