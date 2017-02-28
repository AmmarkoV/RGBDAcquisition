//#include <cv.h>
//#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>


#define MAX_FILENAME_STRING 1024
#define MAX_LINE_CALIBRATION 1024

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

using namespace cv;
using namespace std;


struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char intrinsicParametersSet;
  double intrinsic[9];
  double k1,k2,p1,p2,k3;

  /* CAMERA EXTRINSIC PARAMETERS */
  char extrinsicParametersSet;
  float extrinsicRotationRodriguez[3];
  float extrinsicTranslation[3];
};


int forceUSLocaleToKeepOurSanity()
{
   fprintf(stderr,"Reinforcing EN_US locale\n");
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}



int ReadCalibration(char * filename,struct calibration * calib)
{
  FILE * fp = 0;
  fp = fopen(filename,"r");
  if (fp == 0 ) {  return 0; }

  char line[MAX_LINE_CALIBRATION]={0};

  unsigned int i=0;

  unsigned int category=0;
  unsigned int linesAtCurrentCategory=0;


  while ( fgets(line,MAX_LINE_CALIBRATION,fp)!=0 )
   {
     unsigned int lineLength = strlen ( line );
     if ( lineLength > 0 ) {
                                 if (line[lineLength-1]==10) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                                 if (line[lineLength-1]==13) { line[lineLength-1]=0; /*fprintf(stderr,"-1 newline \n");*/ }
                           }
     if ( lineLength > 1 ) {
                                 if (line[lineLength-2]==10) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                                 if (line[lineLength-2]==13) { line[lineLength-2]=0; /*fprintf(stderr,"-2 newline \n");*/ }
                           }


     if (line[0]=='%') { linesAtCurrentCategory=0; }
     if ( (line[0]=='%') && (line[1]=='I') && (line[2]==0) ) { category=1;    } else
     if ( (line[0]=='%') && (line[1]=='D') && (line[2]==0) ) { category=2;    } else
     if ( (line[0]=='%') && (line[1]=='T') && (line[2]==0) ) { category=3;    } else
     if ( (line[0]=='%') && (line[1]=='R') && (line[2]==0) ) { category=4;    } else
        {
          //TOO MUCH SPAM!
          //fprintf(stderr,"Line %u ( %s ) is category %u lines %u \n",i,line,category,linesAtCurrentCategory);
          if (category==1)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->intrinsic[0] = atof(line); break;
             case 2 :  calib->intrinsic[1] = atof(line); break;
             case 3 :  calib->intrinsic[2] = atof(line); break;
             case 4 :  calib->intrinsic[3] = atof(line); break;
             case 5 :  calib->intrinsic[4] = atof(line); break;
             case 6 :  calib->intrinsic[5] = atof(line); break;
             case 7 :  calib->intrinsic[6] = atof(line); break;
             case 8 :  calib->intrinsic[7] = atof(line); break;
             case 9 :  calib->intrinsic[8] = atof(line); break;
           };
          } else
          if (category==2)
          {
           calib->intrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->k1 = atof(line); break;
             case 2 :  calib->k2 = atof(line); break;
             case 3 :  calib->p1 = atof(line); break;
             case 4 :  calib->p2 = atof(line); break;
             case 5 :  calib->k3 = atof(line); break;
           };
          } else
          if (category==3)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicTranslation[0] = atof(line); break;
             case 2 :  calib->extrinsicTranslation[1] = atof(line); break;
             case 3 :  calib->extrinsicTranslation[2] = atof(line); break;
           };
          } else
          if (category==4)
          {
           calib->extrinsicParametersSet=1;
           switch(linesAtCurrentCategory)
           {
             case 1 :  calib->extrinsicRotationRodriguez[0] = atof(line); break;
             case 2 :  calib->extrinsicRotationRodriguez[1] = atof(line); break;
             case 3 :  calib->extrinsicRotationRodriguez[2] = atof(line); break;
           };
          }


        }

     ++linesAtCurrentCategory;
     ++i;
     line[0]=0;
   }

  return 1;
}


void append_camera_params( const char* out_filename, struct calibration * calib )
{
  char oldFilename[MAX_FILENAME_STRING]={0};
  sprintf(oldFilename,"old%s",out_filename);

    FILE * fp=0;
    fp= fopen(out_filename,"a");
    if (fp==0) { fprintf(stderr,"Could not open output file\n"); return; }

    fprintf( fp, "%%New extrinsic data\n");
    fprintf( fp, "%%Translation T.X, T.Y, T.Z\n");
    fprintf( fp, "%%T\n");
    fprintf( fp, "%f\n",calib->extrinsicTranslation[0]);
    fprintf( fp, "%f\n",calib->extrinsicTranslation[1]);
    fprintf( fp, "%f\n",calib->extrinsicTranslation[2]);

    fprintf( fp, "%%Rotation Vector (Rodrigues) R.X, R.Y, R.Z\n");
    fprintf( fp, "%%R\n");
    fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[0]);
    fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[1]);
    fprintf( fp, "%f\n",calib->extrinsicRotationRodriguez[2]);

   fclose(fp);
}



int calibrateExtrinsicOnly( CvPoint2D32f* image_points_buf, CvSize img_size, CvSize board_size,
                     float square_size, float aspect_ratio,
                     CvMat* camera_matrix, CvMat* dist_coeffs, CvMat** extr_params,
                     CvMat * rot_vects, CvMat * trans_vects )
{
    int code=0;
    int image_count = 1;
    int point_count = board_size.width*board_size.height;
    fprintf(stderr,"Calibrate , image points total %u , images total %u \n",point_count,image_count);
    CvMat* image_points = cvCreateMat( 1, image_count*point_count, CV_32FC2 );
    CvMat* object_points = cvCreateMat( 1, image_count*point_count, CV_32FC3 );
    CvMat* point_counts = cvCreateMat( 1, image_count, CV_32SC1 );
    ;
    int initialPoint=0, j=0, k=0;
    //CvSeqReader reader;

    // initialize arrays of points
    CvPoint2D32f* src_img_pt = (CvPoint2D32f*) image_points_buf;
    CvPoint2D32f* dst_img_pt = ((CvPoint2D32f*)image_points->data.fl) + initialPoint*point_count;
    CvPoint3D32f* obj_pt = ((CvPoint3D32f*)object_points->data.fl) + initialPoint*point_count;

    for( j = 0; j < board_size.height; j++ )
     for( k = 0; k < board_size.width; k++ )
            {
                *obj_pt++ = cvPoint3D32f(j*square_size, k*square_size, 0);
                *dst_img_pt++ = *src_img_pt++;
            }
    cvSet( point_counts, cvScalar(point_count) );

    *extr_params = cvCreateMat( image_count, 6, CV_32FC1 );
    cvGetCols( *extr_params, rot_vects, 0, 3 );
    cvGetCols( *extr_params, trans_vects, 3, 6 );

    cvFindExtrinsicCameraParams2( object_points, image_points,camera_matrix,dist_coeffs,rot_vects, trans_vects);

    code = cvCheckArr( camera_matrix, CV_CHECK_QUIET ) &&
        cvCheckArr( dist_coeffs, CV_CHECK_QUIET ) &&
        cvCheckArr( *extr_params, CV_CHECK_QUIET );



    cvReleaseMat( &object_points );
    cvReleaseMat( &image_points );
    cvReleaseMat( &point_counts );

    return code;

}

int convertRodriguezTo3x3(float * rodriguez , float * result)
{
  if ( (rodriguez==0) ||  (result==0) ) { return 0; }
  float x = rodriguez[0] , y = rodriguez[1] , z = rodriguez[2];
  float th = sqrt( x*x + y*y + z*z );
  float cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  //Switch to control what kind of a result to give :P
  #define PRODUCE_TRANSPOSED_RESULT 0
  // REGULAR  TRANSPOSED
  //  0 1 2     0 3 6
  //  3 4 5     1 4 7
  //  6 7 8     2 5 8

  #if PRODUCE_TRANSPOSED_RESULT
    //TRANSPOSED RESULT
    result[0]=x*x*(1 - cosTh) + cosTh;            result[3]=x*y*(1 - cosTh) - z*sin(th);     result[6]=x*z*(1 - cosTh) + y*sin(th);
    result[1]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;         result[7]=y*z*(1 - cosTh) - x*sin(th);
    result[2]=x*z*(1 - cosTh) - y*sin(th);        result[5]=y*z*(1 - cosTh) + x*sin(th);     result[8]=z*z*(1 - cosTh) + cosTh;
  #else
   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;        result[1]=x*y*(1 - cosTh) - z*sin(th);     result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;       result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;
  #endif

  return 1;
}


int main( int argc, char** argv )
{
    forceUSLocaleToKeepOurSanity();

    if( argc < 2 )
    {
        printf( "This is a camera calibration sample.\n"
            "Usage: calibration\n"
            "     -w <board_width>         # the number of inner corners per one of board dimension\n"
            "     -h <board_height>        # the number of inner corners per another board dimension\n"
            "     [-s <square_size>]       # square size in some user-defined units (1 by default)\n"
            "     -i  # the output filename for image \n"
            "     -c  input/output calibration file\n"
            "\n" );
        return 0;
    }


    CvSize img_size = {640,480};
    CvSize board_size = {6,9};
    CvMat rot_vects, trans_vects;
    float square_size = 1.0 , aspect_ratio = 1.0;
    int viewResult = 0;
    int writeResult = 1;
    int writeResultImage = 0;

    char calibFile[MAX_FILENAME_STRING]={0};
    char imageFile[MAX_FILENAME_STRING]={0};

  int i=0;
  for (i=0; i<argc; i++)
  {
    if (strcmp(argv[i],"-w")==0) { board_size.width=atoi(argv[i+1]);  } else
    if (strcmp(argv[i],"-h")==0) { board_size.height=atoi(argv[i+1]); } else
    //Size of unit
    if (strcmp(argv[i],"-s")==0) { square_size=atof(argv[i+1]);      } else
    if (strcmp(argv[i],"-c")==0) { strncpy(calibFile,argv[i+1],MAX_FILENAME_STRING);      } else
    if (strcmp(argv[i],"-i")==0) { strncpy(imageFile,argv[i+1],MAX_FILENAME_STRING);      } else
    if (strcmp(argv[i],"-v")==0) { viewResult=1;   } else
    if (strcmp(argv[i],"--writeImage")==0) { writeResultImage=1;   } else
    if (strcmp(argv[i],"-n")==0)   { writeResult=0;   }

  }




    IplImage *view = 0, *view_gray = 0;
    int count = 0, found;

    view = cvLoadImage( imageFile, 1 );
    if (view==0)
     {
       fprintf(stderr,"Could not read image file %s \n",imageFile);
       return 1;
     }

    int elem_size = board_size.width*board_size.height*sizeof(CvPoint2D32f);

    CvMemStorage* storage = cvCreateMemStorage( MAX( elem_size*4, 1 << 16 ));
    CvPoint2D32f* image_points_buf = (CvPoint2D32f*)cvAlloc( elem_size );

    if ( (storage==0) || (image_points_buf==0) )
    {
       fprintf(stderr,"Could not allocate OpenCV Buffers for chessboard\n");
       return 2;
    }



    img_size = cvGetSize(view);
    found = cvFindChessboardCorners( view, board_size, image_points_buf, &count, CV_CALIB_CB_ADAPTIVE_THRESH );

    // improve the found corners' coordinate accuracy
    view_gray = cvCreateImage( cvGetSize(view), 8, 1 );
    cvCvtColor( view, view_gray, CV_BGR2GRAY );
     cvShowImage( "Gray View", view_gray );
    cvFindCornerSubPix( view_gray, image_points_buf, count, cvSize(11,11),cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

    cvDrawChessboardCorners( view, board_size, image_points_buf, count, found );
    if ( viewResult )      {  cvShowImage( "Image View", view );          }
    if (writeResultImage)  {  cvSaveImage( "extcalibrate.jpg", view, 0 ); }
    cvReleaseImage( &view_gray );


    struct calibration calib={0};
    ReadCalibration(calibFile,&calib);

    double _dist_coeffs[4]={0}; _dist_coeffs[0]=calib.k1; _dist_coeffs[1]=calib.k2; _dist_coeffs[2]=calib.p1; _dist_coeffs[3]=calib.p2;
    CvMat camera = cvMat( 3, 3, CV_64F, calib.intrinsic );
    CvMat dist_coeffs = cvMat( 1, 4, CV_64F, _dist_coeffs );
    CvMat *extr_params = 0;
    //CvMat *reproj_errs = 0;
    //double avg_reproj_err = 0;

    int code = calibrateExtrinsicOnly( image_points_buf, img_size, board_size, square_size, aspect_ratio, &camera, &dist_coeffs, &extr_params, &rot_vects, &trans_vects );
    fprintf(stderr,".. code %u .. \n",code);


    fprintf( stderr, " Rot : %f ",rot_vects.data.fl[0]); fprintf( stderr, "%f ",rot_vects.data.fl[1]); fprintf( stderr, "%f\n",rot_vects.data.fl[2]);
    fprintf( stderr, " Tra : %f ",trans_vects.data.fl[0]); fprintf( stderr, "%f ",trans_vects.data.fl[1]); fprintf( stderr, "%f\n",trans_vects.data.fl[2]);

    calib.extrinsicRotationRodriguez[0]=rot_vects.data.fl[0];
    calib.extrinsicRotationRodriguez[1]=rot_vects.data.fl[1];
    calib.extrinsicRotationRodriguez[2]=rot_vects.data.fl[2];

    calib.extrinsicTranslation[0]=trans_vects.data.fl[0];
    calib.extrinsicTranslation[1]=trans_vects.data.fl[1];
    calib.extrinsicTranslation[2]=trans_vects.data.fl[2];


    if (writeResult) { append_camera_params(calibFile,&calib); }

    float result[9]={0};
    convertRodriguezTo3x3(calib.extrinsicRotationRodriguez,(float*) &result);
    fprintf( stderr, "  %f ",result[0]); fprintf( stderr, "%f ",result[1]); fprintf( stderr, "%f\n",result[2]);
    fprintf( stderr, "  %f ",result[3]); fprintf( stderr, "%f ",result[4]); fprintf( stderr, "%f\n",result[5]);
    fprintf( stderr, "  %f ",result[6]); fprintf( stderr, "%f ",result[7]); fprintf( stderr, "%f\n",result[8]);

    fprintf(stderr,"\n\n\n");

    float out[9]={0};
    CvMat rotM = cvMat( 3, 3, CV_32F , out );
    cvRodrigues2(&rot_vects,&rotM);
    fprintf( stderr, "  %f ",rotM.data.fl[0]); fprintf( stderr, "%f ",rotM.data.fl[1]); fprintf( stderr, "%f\n",rotM.data.fl[2]);
    fprintf( stderr, "  %f ",rotM.data.fl[3]); fprintf( stderr, "%f ",rotM.data.fl[4]); fprintf( stderr, "%f\n",rotM.data.fl[5]);
    fprintf( stderr, "  %f ",rotM.data.fl[6]); fprintf( stderr, "%f ",rotM.data.fl[7]); fprintf( stderr, "%f\n",rotM.data.fl[8]);


    if ( viewResult ) { cvWaitKey(0); }

    return 0;
}
