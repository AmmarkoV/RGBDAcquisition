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


using namespace cv;


struct sgbmCalibrationStuff
{
  CvMat* _M1,*_M2,*_D1,*_D2,*_R1,*_R2,*_P1,*_P2,*_R,*_T,*_Q;
  cv::Mat M1,M2,D1,D2,R1,R2,P1,P2,R,T,Q;
  int calibrationIsSet;
};

struct sgbmCalibrationStuff sgbmCalib;


int oldKindOfDisplayCalibrationReading(char * disparityCalibrationPath)
{
  if (disparityCalibrationPath==0) { return 0; }


// OpenCV 3.0  might require something like this sgbmCalib.M1 = cv::Mat(sgbmCalib._M1->rows, sgbmCalib._M1->cols, sgbmCalib._M1->type, sgbmCalib._M1->data.*);
  char filenameOfFile[1024]={0};
  snprintf(filenameOfFile,1024,"%s/M1.xml",disparityCalibrationPath);
  sgbmCalib._M1 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.M1 = cv::Mat(sgbmCalib._M1);

  snprintf(filenameOfFile,1024,"%s/M2.xml",disparityCalibrationPath);
  sgbmCalib._M2 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.M2 = cv::Mat(sgbmCalib._M2);

  snprintf(filenameOfFile,1024,"%s/D1.xml",disparityCalibrationPath);
  sgbmCalib._D1 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.D1 = cv::Mat(sgbmCalib._D1);

  snprintf(filenameOfFile,1024,"%s/D2.xml",disparityCalibrationPath);
  sgbmCalib._D2 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.D2 = cv::Mat(sgbmCalib._D2);

  snprintf(filenameOfFile,1024,"%s/R1.xml",disparityCalibrationPath);
  sgbmCalib._R1 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.R1 = cv::Mat(sgbmCalib._R1);

  snprintf(filenameOfFile,1024,"%s/R2.xml",disparityCalibrationPath);
  sgbmCalib._R2 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.R2 = cv::Mat(sgbmCalib._R2);

  snprintf(filenameOfFile,1024,"%s/P1.xml",disparityCalibrationPath);
  sgbmCalib._P1 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.P1 = cv::Mat(sgbmCalib._P1);

  snprintf(filenameOfFile,1024,"%s/P2.xml",disparityCalibrationPath);
  sgbmCalib._P2 = (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.P2 = cv::Mat(sgbmCalib._P2);

  snprintf(filenameOfFile,1024,"%s/R.xml",disparityCalibrationPath);
  sgbmCalib._R =  (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.R = cv::Mat(sgbmCalib._R);

  snprintf(filenameOfFile,1024,"%s/T.xml",disparityCalibrationPath);
  sgbmCalib._T =  (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.T = cv::Mat(sgbmCalib._T);

  snprintf(filenameOfFile,1024,"%s/Q.xml",disparityCalibrationPath);
  sgbmCalib._Q =  (CvMat*) cvLoad(filenameOfFile);
  sgbmCalib.Q = cv::Mat(sgbmCalib._Q);

  sgbmCalib.calibrationIsSet=1;

 return 1;
}



int newKindOfDisplayCalibrationReading(char * disparityCalibrationPath)
{
 fprintf(stderr,"newKindOfDisplayCalibrationReading(%s)\n",disparityCalibrationPath);
 char intrinsic_filename[2048]={0};
 char extrinsic_filename[2048]={0};

 snprintf(intrinsic_filename,2048,"%s/intrinsics.yml",disparityCalibrationPath);
 snprintf(extrinsic_filename,2048,"%s/extrinsics.yml",disparityCalibrationPath);

       FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
        if(!fs.isOpened())
        {
            printf("Failed to open intrinsics file %s\n", intrinsic_filename);
            return -1;
        }

        fs["M1"] >> sgbmCalib.M1;
        fs["D1"] >> sgbmCalib.D1;
        fs["M2"] >> sgbmCalib.M2;
        fs["D2"] >> sgbmCalib.D2;

        fs.open(extrinsic_filename, CV_STORAGE_READ);
        if(!fs.isOpened())
        {
            printf("Failed to open extrinsics file %s\n", extrinsic_filename);
            return -1;
        }

        fs["R"] >> sgbmCalib.R;
        fs["T"] >> sgbmCalib.T;
        std::cout << "R : \n"<<sgbmCalib.R << "\n";
        std::cout << "T : \n"<<sgbmCalib.T << "\n";

        fs["R1"] >> sgbmCalib.R1;
        fs["R2"] >> sgbmCalib.R2;
        fs["P1"] >> sgbmCalib.P1;
        fs["P2"] >> sgbmCalib.P2;
        fs["Q"] >> sgbmCalib.Q;


  sgbmCalib.calibrationIsSet=1;
  return 1;
}



int doSGBM( cv::Mat *leftBGR,cv::Mat *rightBGR , unsigned int SADWindowSize ,  unsigned int speckleRange, char * disparityCalibrationPath)
{
    char* intrinsic_filename = disparityCalibrationPath;
    char* extrinsic_filename = disparityCalibrationPath;
    if (!sgbmCalib.calibrationIsSet) { intrinsic_filename = 0;  extrinsic_filename = 0; }
    const char* disparity_filename = 0;
    const char* point_cloud_filename = 0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    StereoBM bm;
    StereoSGBM sgbm;
    StereoVar var;



    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 ,img2 ;
    img1 = *leftBGR;
    img2 = *rightBGR;

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

    if( sgbmCalib.calibrationIsSet )
    {
        stereoRectify( sgbmCalib.M1, sgbmCalib.D1, sgbmCalib.M2, sgbmCalib.D2, img_size, sgbmCalib.R, sgbmCalib.T, sgbmCalib.R1, sgbmCalib.R2, sgbmCalib.P1, sgbmCalib.P2, sgbmCalib.Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(sgbmCalib.M1, sgbmCalib.D1, sgbmCalib.R1, sgbmCalib.P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(sgbmCalib.M2, sgbmCalib.D2, sgbmCalib.R2, sgbmCalib.P2, img_size, CV_16SC2, map21, map22);

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
    bm.state->speckleRange = speckleRange;
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
    if( alg == STEREO_BM )  { bm(img1, img2, disp);  } else
    if( alg == STEREO_VAR ) { var(img1, img2, disp); } else
    if( alg == STEREO_SGBM || alg == STEREO_HH )
                            { sgbm(img1, img2, disp); }

    t = getTickCount() - t;
    printf("OpenCV Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR ) { disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.)); } else
                            { disp.convertTo(disp8, CV_8U); }




    cv::Point pt1=cv::Point(0,0);
    cv::Point pt2=cv::Point(img1.cols,0);
    cv::Scalar color=cv::Scalar(0,255,0);
    unsigned int i=0;
    unsigned int blockY=(unsigned int) img1.rows/15;
    for (i=0; i<img1.rows/15; i++)
    {
       pt1.y=i*blockY; pt2.y=i*blockY;
       cv::line(img1,pt1,pt2,   color, 1, 8, 0);
       cv::line(img2,pt1,pt2,   color, 1, 8, 0);
    }

    img1.copyTo(*leftBGR);
    img2.copyTo(*rightBGR);


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
        { imwrite(disparity_filename, disp8); }

    if(point_cloud_filename)
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, sgbmCalib.Q, true);
        printf("\n");
    }


 return 1;
}
