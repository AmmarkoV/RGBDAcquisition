#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;





int calicrateExtrinsicOnly( CvSeq* image_points_seq, CvSize img_size, CvSize board_size,
                     float square_size, float aspect_ratio,
                     CvMat* camera_matrix, CvMat* dist_coeffs, CvMat** extr_params,
                     CvMat** reproj_errs, double* avg_reproj_err )
{

    int code;
    int image_count = image_points_seq->total;
    int point_count = board_size.width*board_size.height;
    CvMat* image_points = cvCreateMat( 1, image_count*point_count, CV_32FC2 );
    CvMat* object_points = cvCreateMat( 1, image_count*point_count, CV_32FC3 );
    CvMat* point_counts = cvCreateMat( 1, image_count, CV_32SC1 );
    CvMat rot_vects, trans_vects;
    int i, j, k;
    CvSeqReader reader;
    cvStartReadSeq( image_points_seq, &reader );

    // initialize arrays of points
    for( i = 0; i < image_count; i++ )
    {
        CvPoint2D32f* src_img_pt = (CvPoint2D32f*)reader.ptr;
        CvPoint2D32f* dst_img_pt = ((CvPoint2D32f*)image_points->data.fl) + i*point_count;
        CvPoint3D32f* obj_pt = ((CvPoint3D32f*)object_points->data.fl) + i*point_count;

        for( j = 0; j < board_size.height; j++ )
            for( k = 0; k < board_size.width; k++ )
            {
                *obj_pt++ = cvPoint3D32f(j*square_size, k*square_size, 0);
                *dst_img_pt++ = *src_img_pt++;
            }
        CV_NEXT_SEQ_ELEM( image_points_seq->elem_size, reader );
    }

    cvSet( point_counts, cvScalar(point_count) );

    *extr_params = cvCreateMat( image_count, 6, CV_32FC1 );
    cvGetCols( *extr_params, &rot_vects, 0, 3 );
    cvGetCols( *extr_params, &trans_vects, 3, 6 );

    cvZero( camera_matrix );
    cvZero( dist_coeffs );

    cvCalibrateCamera2( object_points, image_points, point_counts,
                        img_size, camera_matrix, dist_coeffs,
                        &rot_vects, &trans_vects, 0 );


    cvFindExtrinsicCameraParams2( object_points, image_points,camera_matrix,dist_coeffs,&rot_vects, &trans_vects);

    code = cvCheckArr( camera_matrix, CV_CHECK_QUIET ) &&
        cvCheckArr( dist_coeffs, CV_CHECK_QUIET ) &&
        cvCheckArr( *extr_params, CV_CHECK_QUIET );



    fprintf( stderr, " Rot : %f\n",rot_vects.data.fl[0]); fprintf( stderr, "%f\n",rot_vects.data.fl[1]); fprintf( stderr, "%f\n",rot_vects.data.fl[2]);
    fprintf( stderr, " Tra : %f\n",trans_vects.data.fl[0]); fprintf( stderr, "%f\n",trans_vects.data.fl[1]); fprintf( stderr, "%f\n",trans_vects.data.fl[2]);


    cvReleaseMat( &object_points );
    cvReleaseMat( &image_points );
    cvReleaseMat( &point_counts );

    return code;

}






int main( int argc, char** argv )
{
    CvSize img_size = {640,480};
    CvSize board_size = {6,9};
    float square_size = 1.0 , aspect_ratio = 1.0;

    IplImage *view = 0, *view_gray = 0;
    int count = 0, found;

    view = cvLoadImage( argv[1], 1 );

    int elem_size = board_size.width*board_size.height*sizeof(CvPoint2D32f);

    CvMemStorage* storage = cvCreateMemStorage( MAX( elem_size*4, 1 << 16 ));
    CvPoint2D32f* image_points_buf = (CvPoint2D32f*)cvAlloc( elem_size );
    CvSeq* image_points_seq = cvCreateSeq( 0, sizeof(CvSeq), elem_size, storage );

    img_size = cvGetSize(view);
    found = cvFindChessboardCorners( view, board_size, image_points_buf, &count, CV_CALIB_CB_ADAPTIVE_THRESH );

    // improve the found corners' coordinate accuracy
    view_gray = cvCreateImage( cvGetSize(view), 8, 1 );
    cvCvtColor( view, view_gray, CV_BGR2GRAY );
    cvFindCornerSubPix( view_gray, image_points_buf, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
    cvReleaseImage( &view_gray );

    cvDrawChessboardCorners( view, board_size, image_points_buf, count, found );
    cvShowImage( "Image View", view );



    double _camera[9], _dist_coeffs[4];
    CvMat camera = cvMat( 3, 3, CV_64F, _camera );
    CvMat dist_coeffs = cvMat( 1, 4, CV_64F, _dist_coeffs );
    CvMat *extr_params = 0, *reproj_errs = 0;
    double avg_reproj_err = 0;

    int code = calicrateExtrinsicOnly( image_points_seq, img_size, board_size, square_size, aspect_ratio, &camera, &dist_coeffs, &extr_params, &reproj_errs, &avg_reproj_err );


     waitKey(0);


    return 0;
}
