#pragma warning( disable: 4996 )
/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warrenty, support or any guarentee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OTHER OPENCV SITES:
   * The source code is on sourceforge at:
     http://sourceforge.net/projects/opencvlibrary/
   * The OpenCV wiki page (As of Oct 1, 2008 this is down for changing over servers, but should come back):
     http://opencvlibrary.sourceforge.net/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
   ************************************************** */

/*
	Modified by Martin Peris Martorell (info@martinperis.com) in order to accept some configuration
	parameters and store all the calibration data as xml files.

*/

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
#include "stereo_calib.h"

using namespace std;

    int haveInitialization=0;
    unsigned int goodCalibrationFrames=0;

    int displayCorners = 1;
    int showUndistorted = 1;
    bool isVerticalStereo = false;//OpenCV can handle left-right
    //or up-down camera arrangements
    const int maxScale = 1;
    float squareSize = 0; //Chessboard square size in cm
    vector<string> imageNames[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<int> npoints;
    vector<uchar> active[2];
    CvSize imageSize = {0,0};
    // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    double Q[4][4];
    CvMat _M1,_M2,_D1,_D2,_R,_T,_E,_F,_Q;


void dbg(int line,char * fileName)
{
    fprintf (stderr,"line : %d of file \"%s\".\n", line, fileName);
}


int initializeCalibration(cv::Mat * leftImgRGB ,
                          cv::Mat * rightImgRGB )
{
     _M1 = cvMat(3, 3, CV_64F, M1 );
     _M2 = cvMat(3, 3, CV_64F, M2 );
     _D1 = cvMat(1, 5, CV_64F, D1 );
     _D2 = cvMat(1, 5, CV_64F, D2 );
     _R = cvMat(3, 3, CV_64F, R );
     _T = cvMat(3, 1, CV_64F, T );
     _E = cvMat(3, 3, CV_64F, E );
     _F = cvMat(3, 3, CV_64F, F );
     _Q = cvMat(4,4, CV_64F, Q);

    haveInitialization=1;
    return 1;
}




int stopAppending( char * filenameFinal )
{
  FILE * xmlFileOutput = fopen (filenameFinal,"a");
  if (xmlFileOutput!=NULL)
  {
    fprintf(xmlFileOutput,"</imagelist>\n");
    fprintf(xmlFileOutput,"</opencv_storage>\n\n");

    fclose (xmlFileOutput);
    return 1;
  }
 return 0;
}


int finalizeCalibration(char * outputFolder,int nx, int ny, float _squareSize)
{
  char filenameFinal[1024]={0};
  snprintf(filenameFinal,1024,"%s/stereo_calib.xml",outputFolder);
  stopAppending(filenameFinal);

  char outputIntrinsics[1024]={0};
  char outputExtrinsics[1024]={0};
  snprintf(outputIntrinsics,1024,"%s/intrinsics.yml",outputFolder);
  snprintf(outputExtrinsics,1024,"%s/extrinsics.yml",outputFolder);



  fprintf(stderr,"Now running part 2 of calibration code..!\n");
  fprintf(stderr,"This might take a while( hang on.. ) \n");
   stereoCalibMain(outputIntrinsics,outputExtrinsics,filenameFinal ,nx, ny,_squareSize);
  fprintf(stderr,"done..\n");
  haveInitialization=0;
 return 1;
}


int appendImages(
                  char * outputFolder,
                  cv::Mat * leftImgRGB ,
                  cv::Mat * rightImgRGB,
                  unsigned int frameNumber
                )
{
 char filenameFinal[1024]={0};
 snprintf(filenameFinal,1024,"%s/left_%05u.jpg",outputFolder,frameNumber);
 imwrite(filenameFinal,*leftImgRGB);

 snprintf(filenameFinal,1024,"%s/right_%05u.jpg",outputFolder,frameNumber);
 imwrite(filenameFinal,*rightImgRGB);

 snprintf(filenameFinal,1024,"%s/stereo_calib.xml",outputFolder);
 FILE * xmlFileOutput;
 if (frameNumber==0)
 {
  xmlFileOutput = fopen (filenameFinal,"w");
  if (xmlFileOutput!=NULL)
  {
    fprintf(xmlFileOutput,"<?xml version=\"1.0\"?>\n");
    fprintf(xmlFileOutput,"<opencv_storage>\n");
    fprintf(xmlFileOutput,"<imagelist>\n");
    fprintf(xmlFileOutput,"\"%s/left_%05u.jpg\"\n",outputFolder,frameNumber);
    fprintf(xmlFileOutput,"\"%s/right_%05u.jpg\"\n",outputFolder,frameNumber);
    fclose (xmlFileOutput);
  }
 } else
 {
  xmlFileOutput = fopen (filenameFinal,"a");
  if (xmlFileOutput!=NULL)
  {
    fprintf(xmlFileOutput,"\"%s/left_%05u.jpg\"\n",outputFolder,frameNumber);
    fprintf(xmlFileOutput,"\"%s/right_%05u.jpg\"\n",outputFolder,frameNumber);
    fclose (xmlFileOutput);
  }
 }




}


//
// Given a list of chessboard images, the number of corners (nx, ny)
// on the chessboards, and a flag: useCalibrated for calibrated (0) or
// uncalibrated (1: use cvStereoCalibrate(), 2: compute fundamental
// matrix separately) stereo. Calibrate the cameras and display the
// rectified results along with the computed disparity images.
//
static void StereoCalib(cv::Mat * leftImgRGB ,
                        cv::Mat * rightImgRGB,
                        cv::Mat * leftImgGray ,
                        cv::Mat * rightImgGray,
                        int nx, int ny, int useUncalibrated, float _squareSize,
                        char * disparityCalibrationOutputPath
                        )
{
    int i, j, lr, nframes, n = nx*ny, N = 0;
    vector<CvPoint2D32f> LeftPoints(n);
    vector<CvPoint2D32f> RightPoints(n);
    squareSize = _squareSize;


   if(!haveInitialization)
   {
        initializeCalibration(leftImgRGB,rightImgRGB);
   }

        // READ IN THE LIST OF CHESSBOARDS:
        int count = 0, chessbordFoundL=0, chessbordFoundR=0;

         // UGLY HACK
        IplImage imgconvLeft=*leftImgRGB;
        IplImage imgconvRight=*rightImgRGB;
        IplImage imgconvGLeft=*leftImgGray;
        IplImage imgconvGRight=*rightImgGray;

        IplImage* imgL=&imgconvLeft;
        IplImage* imgR=&imgconvRight;
        IplImage* imgGL=&imgconvGLeft;
        IplImage* imgGR=&imgconvGRight;

        imageSize = cvGetSize(imgL);
        //FIND CHESSBOARDS AND CORNERS THEREIN:

        if ( cvFindChessboardCorners( imgL, cvSize(nx, ny), &LeftPoints[0], &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE) )
        {
          cvFindCornerSubPix( imgGL, &LeftPoints[0], count, cvSize(11, 11), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01) );
          chessbordFoundL=1;
        }

        if ( cvFindChessboardCorners( imgR, cvSize(nx, ny), &RightPoints[0], &count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE) )
        {
          cvFindCornerSubPix( imgGR, &RightPoints[0], count, cvSize(11, 11), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01) );
          chessbordFoundR=1;
        }

        //This has to be done before we draw on the images and we ruin them..!
        if ( (chessbordFoundL) && (chessbordFoundR) )
        {
          appendImages( disparityCalibrationOutputPath,
                        leftImgRGB ,
                        rightImgRGB,
                        goodCalibrationFrames
                      );
           ++goodCalibrationFrames;
        }


        cvDrawChessboardCorners( imgL, cvSize(nx, ny), &LeftPoints[0],  count, chessbordFoundL );
        cvDrawChessboardCorners( imgR, cvSize(nx, ny), &RightPoints[0],  count, chessbordFoundR );

  return;

  /*
        vector<CvPoint2D32f>& pts = points[lr];
  dbg(__LINE__,__FILE__);

        N = pts.size();
        pts.resize(N + n, cvPoint2D32f(0,0));
        active[lr].push_back((uchar)result);
        //assert( result != 0 );
        if( result )
        {
            //Calibration will suffer without subpixel interpolation
            cvFindCornerSubPix( img, &temp[0], count,
                                cvSize(11, 11), cvSize(-1,-1),
                                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                               30, 0.01) );
            copy( temp.begin(), temp.end(), pts.begin() + N );
        }
        cvReleaseImage( &img );
    }

  dbg(__LINE__,__FILE__);

// HARVEST CHESSBOARD 3D OBJECT POINT LIST:
    nframes = active[0].size();//Number of good chessboads found
    objectPoints.resize(nframes*n);
    for( i = 0; i < ny; i++ )
        for( j = 0; j < nx; j++ )
            objectPoints[i*nx + j] = cvPoint3D32f(i*squareSize, j*squareSize, 0);
    for( i = 1; i < nframes; i++ )
        copy( objectPoints.begin(), objectPoints.begin() + n, objectPoints.begin() + i*n );
    npoints.resize(nframes,n);
    N = nframes*n;
    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);

  dbg(__LINE__,__FILE__);

// CALIBRATE THE STEREO CAMERAS
    printf("Running stereo calibration ...");
    fflush(stdout);
    cvStereoCalibrate( &_objectPoints, &_imagePoints1,
                       &_imagePoints2, &_npoints,
                       &_M1, &_D1, &_M2, &_D2,
                       imageSize, &_R, &_T, &_E, &_F,
                       cvTermCriteria(CV_TERMCRIT_ITER+
                                      CV_TERMCRIT_EPS, 100, 1e-5),
                       CV_CALIB_FIX_ASPECT_RATIO +
                       CV_CALIB_ZERO_TANGENT_DIST +
                       CV_CALIB_SAME_FOCAL_LENGTH );
    printf(" done\n");

      dbg(__LINE__,__FILE__);

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    vector<CvPoint3D32f> lines[2];
    points[0].resize(N);
    points[1].resize(N);
    _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    lines[0].resize(N);
    lines[1].resize(N);
    CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
    CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
//Always work in undistorted space
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,
                       &_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,
                       &_M2, &_D2, 0, &_M2 );
    cvComputeCorrespondEpilines( &_imagePoints1, 1, &_F, &_L1 );
    cvComputeCorrespondEpilines( &_imagePoints2, 2, &_F, &_L2 );
    double avgErr = 0;
    for( i = 0; i < N; i++ )
    {
        double err = fabs(points[0][i].x*lines[1][i].x +
                          points[0][i].y*lines[1][i].y + lines[1][i].z)
                     + fabs(points[1][i].x*lines[0][i].x +
                            points[1][i].y*lines[0][i].y + lines[0][i].z);
        avgErr += err;
    }
    printf( "avg err = %g\n", avgErr/(nframes*n) );

  dbg(__LINE__,__FILE__);

//COMPUTE AND DISPLAY RECTIFICATION
    if( showUndistorted )
    {
        CvMat* mx1 = cvCreateMat( imageSize.height, imageSize.width, CV_32F );
        CvMat* my1 = cvCreateMat( imageSize.height, imageSize.width, CV_32F );
        CvMat* mx2 = cvCreateMat( imageSize.height, imageSize.width, CV_32F );
        CvMat* my2 = cvCreateMat( imageSize.height, imageSize.width, CV_32F );
        CvMat* img1r = cvCreateMat( imageSize.height, imageSize.width, CV_8U );
        CvMat* img2r = cvCreateMat( imageSize.height, imageSize.width, CV_8U );
        CvMat* disp = cvCreateMat( imageSize.height, imageSize.width, CV_16S );
        CvMat* vdisp = cvCreateMat( imageSize.height, imageSize.width, CV_8U );
        CvMat* pair;
        double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
        CvMat _R1 = cvMat(3, 3, CV_64F, R1);
        CvMat _R2 = cvMat(3, 3, CV_64F, R2);
// IF BY CALIBRATED (BOUGUET'S METHOD)
        if( useUncalibrated == 0 )
        {
            CvMat _P1 = cvMat(3, 4, CV_64F, P1);
            CvMat _P2 = cvMat(3, 4, CV_64F, P2);
            cvStereoRectify( &_M1, &_M2, &_D1, &_D2, imageSize, &_R, &_T, &_R1, &_R2, &_P1, &_P2, &_Q, 0); //CV_CALIB_ZERO_DISPARITY
            isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);
            //Precompute maps for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_P1,mx1,my1);
            cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_P2,mx2,my2);

            //Save parameters
            cvSave("M1.xml",&_M1);
            cvSave("D1.xml",&_D1);
            cvSave("R1.xml",&_R1);
            cvSave("P1.xml",&_P1);
            cvSave("M2.xml",&_M2);
            cvSave("D2.xml",&_D2);
            cvSave("R2.xml",&_R2);
            cvSave("P2.xml",&_P2);
            cvSave("R.xml",&_R);
            cvSave("T.xml",&_T);
            cvSave("Q.xml",&_Q);
            cvSave("mx1.xml",mx1);
            cvSave("my1.xml",my1);
            cvSave("mx2.xml",mx2);
            cvSave("my2.xml",my2);

        }
//OR ELSE HARTLEY'S METHOD
        else if( useUncalibrated == 1 || useUncalibrated == 2 )
            // use intrinsic parameters of each camera, but
            // compute the rectification transformation directly
            // from the fundamental matrix
        {
            double H1[3][3], H2[3][3], iM[3][3];
            CvMat _H1 = cvMat(3, 3, CV_64F, H1);
            CvMat _H2 = cvMat(3, 3, CV_64F, H2);
            CvMat _iM = cvMat(3, 3, CV_64F, iM);
            //Just to show you could have independently used F
            if( useUncalibrated == 2 )
                cvFindFundamentalMat( &_imagePoints1, &_imagePoints2, &_F);
            cvStereoRectifyUncalibrated( &_imagePoints1, &_imagePoints2, &_F, imageSize, &_H1, &_H2, 3);
            cvInvert(&_M1, &_iM);
            cvMatMul(&_H1, &_M1, &_R1);
            cvMatMul(&_iM, &_R1, &_R1);
            cvInvert(&_M2, &_iM);
            cvMatMul(&_H2, &_M2, &_R2);
            cvMatMul(&_iM, &_R2, &_R2);
            //Precompute map for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);

            cvInitUndistortRectifyMap(&_M2,&_D1,&_R2,&_M2,mx2,my2);
        }
        else
            assert(0);
        cvNamedWindow( "rectified", 1 );
// RECTIFY THE IMAGES AND FIND DISPARITY MAPS
        if( !isVerticalStereo )
            pair = cvCreateMat( imageSize.height, imageSize.width*2, CV_8UC3 );
        else
            pair = cvCreateMat( imageSize.height*2, imageSize.width, CV_8UC3 );
//Setup for finding stereo corrrespondences
        CvStereoBMState *BMState = cvCreateStereoBMState();
        assert(BMState != 0);
        BMState->preFilterSize=41;
        BMState->preFilterCap=31;
        BMState->SADWindowSize=41;
        BMState->minDisparity=-64;
        BMState->numberOfDisparities=128;
        BMState->textureThreshold=10;
        BMState->uniquenessRatio=15;
        for( i = 0; i < nframes; i++ )
        {
            IplImage* img1=0;//=cvLoadImage(imageNames[0][i].c_str(),0);
            IplImage* img2=0;//=cvLoadImage(imageNames[1][i].c_str(),0);
            if( img1 && img2 )
            {
                CvMat part;
                cvRemap( img1, img1r, mx1, my1 );
                cvRemap( img2, img2r, mx2, my2 );
                if( !isVerticalStereo || useUncalibrated != 0 )
                {
                    // When the stereo camera is oriented vertically,
                    // useUncalibrated==0 does not transpose the
                    // image, so the epipolar lines in the rectified
                    // images are vertical. Stereo correspondence
                    // function does not support such a case.
                    cvFindStereoCorrespondenceBM( img1r, img2r, disp,
                                                  BMState);
                    cvNormalize( disp, vdisp, 0, 256, CV_MINMAX );
                    cvNamedWindow( "disparity" );
                    cvShowImage( "disparity", vdisp );
                }
                if( !isVerticalStereo )
                {
                    cvGetCols( pair, &part, 0, imageSize.width );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetCols( pair, &part, imageSize.width, imageSize.width*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < imageSize.height; j += 16 )
                        cvLine( pair, cvPoint(0,j), cvPoint(imageSize.width*2,j), CV_RGB(0,255,0));
                }
                else
                {
                    cvGetRows( pair, &part, 0, imageSize.height );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetRows( pair, &part, imageSize.height,
                               imageSize.height*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < imageSize.width; j += 16 )
                        cvLine( pair, cvPoint(j,0), cvPoint(j,imageSize.height*2), CV_RGB(0,255,0));
                }
                cvShowImage( "rectified", pair );
                if( cvWaitKey(3) == 27 )
                    break;
            }
            cvReleaseImage( &img1 );
            cvReleaseImage( &img2 );
        }
        cvReleaseStereoBMState(&BMState);
        cvReleaseMat( &mx1 );
        cvReleaseMat( &my1 );
        cvReleaseMat( &mx2 );
        cvReleaseMat( &my2 );
        cvReleaseMat( &img1r );
        cvReleaseMat( &img2r );
        cvReleaseMat( &disp );
    }
    */
}




int doCalibrationStep(cv::Mat * leftImgRGB ,
                      cv::Mat * rightImgRGB ,
                        cv::Mat * leftImgGray ,
                        cv::Mat * rightImgGray,
                      unsigned int horizontalSquares,unsigned int verticalSquares,float calibSquareSize,char * disparityCalibrationOutputPath)
{
  StereoCalib(leftImgRGB,rightImgRGB,leftImgGray,rightImgGray, horizontalSquares, verticalSquares, 0 , calibSquareSize,disparityCalibrationOutputPath);
  return 1;
}
