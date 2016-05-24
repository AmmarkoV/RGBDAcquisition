#include "reconstruction.h"
#include "homography.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

struct Point2D
{
  float x , y;
};


struct Point2DCorrespondance
{
  struct Point2D  * listSource;
  struct Point2D  * listTarget;
  unsigned int listCurrent;
  unsigned int listMax;
};

int getPointListNumber(const char * filenameLeft )
{
  fprintf(stderr,"reconstruct3D(%s)\n",filenameLeft);

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filenameLeft,"r");
  if (fp!=0)
  {

    char * line = NULL;
    size_t len = 0;
    unsigned int numberOfLines=0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        ++numberOfLines;
    }

    fclose(fp);
    if (line) { free(line); }
    return numberOfLines;
  }

 return 0;
}



static float distance3D(float p1X , float p1Y  , float p1Z ,  float p2X , float p2Y , float p2Z)
{
  float vect_x = p1X - p2X;
  float vect_y = p1Y - p2Y;
  float vect_z = p1Z - p2Z;
  float len = sqrt( vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);
  if(len == 0) len = 1.0f;
return len;
}

struct Point2DCorrespondance * readPointList(const char * filenameLeft )
{

   struct Point2DCorrespondance * newList=0;


   newList = (struct Point2DCorrespondance *) malloc( sizeof ( struct Point2DCorrespondance )  );

   newList->listMax  =  getPointListNumber(filenameLeft);
   newList->listSource = (struct Point2D *) malloc( sizeof ( struct Point2D ) * newList->listMax  );
   newList->listTarget = (struct Point2D *) malloc( sizeof ( struct Point2D ) * newList->listMax  );
   newList->listCurrent=0;

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filenameLeft,"r");
  if (fp!=0)
  {

    char * line = NULL;
    char * lineStart = line;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        lineStart = line;
        while (*lineStart==' ') { ++lineStart; }

        //printf("Retrieved line of length %zu :\n", read);
        //printf("%s", lineStart);


        char * num1 = lineStart; // number1 start to first ' '

        char * num2 = strchr(num1 , ' ');
        while (*num2==' ') { *num2=0; ++num2; }

        char * num3 = strchr(num2 , ' ');
        while (*num3==' ') { *num3=0; ++num3; }

        char * num4 = strchr(num3, ' ');
        while (*num4==' ') { *num4=0; ++num4; }

        //printf("vals are |%s|%s|%s|%s| \n", num1,num2,num3,num4);
        //printf("floats are |%0.2f|%0.2f|%0.2f|%0.2f| \n",atof(num1),atof(num2),atof(num3),atof(num4));

        newList->listSource[newList->listCurrent].x = atof(num1);
        newList->listSource[newList->listCurrent].y = atof(num2);
        newList->listTarget[newList->listCurrent].x = atof(num3);
        newList->listTarget[newList->listCurrent].y = atof(num4);
        ++newList->listCurrent;
    }

    fclose(fp);
    if (line) { free(line); }
    return newList;
}

fprintf(stderr,"Done.. \n");
return 0;
}





int  reconstruct3D(const char * filenameLeft , unsigned int useOpenCVEstimator )
{
  unsigned int s=2;
  char filename[512]={0};




  snprintf(filename,512,"%s/%s1.jpg",filenameLeft ,filenameLeft);
  cv::Mat image1 = cv::imread(filename  , CV_LOAD_IMAGE_COLOR);
  if(! image1.data ) { fprintf(stderr,"Image1 missing \n"); return 0; }

  snprintf(filename,512,"%s/%s2.jpg",filenameLeft ,filenameLeft);
  cv::Mat image2 = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if(! image2.data ) { fprintf(stderr,"Image2 missing \n"); return 0; }

  cv::Mat imageOutput = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  cv::rectangle( imageOutput,
                 cv::Point( 0  , 0 ),
                 cv::Point( imageOutput.cols  , imageOutput.rows ),
                 cv::Scalar( 0, 0, 0 ),
                 -1,
                 8
              );

  std::vector<cv::Point2f> srcPoints;
  std::vector<cv::Point2f> dstPoints;

  snprintf(filename,512,"%s/%s_matches.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * correspondances = readPointList( filename );
  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {

      printf("#%u (%0.2f , %0.2f ) -> ( %0.2f , %0.2f) \n" , i ,
        correspondances->listSource[i].x ,
        correspondances->listSource[i].y ,
        correspondances->listTarget[i].x ,
        correspondances->listTarget[i].y  );



      cv::Point2f srcPT; srcPT.x = correspondances->listSource[i].x; srcPT.y = correspondances->listSource[i].y;
      cv::Point2f dstPT; dstPT.x = correspondances->listTarget[i].x; dstPT.y = correspondances->listTarget[i].y;
      srcPoints.push_back(srcPT);
      dstPoints.push_back(dstPT);

      cv::rectangle( image1,
                     cv::Point( correspondances->listSource[i].x-s  , correspondances->listSource[i].y-s ),
                     cv::Point( correspondances->listSource[i].x+s  , correspondances->listSource[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );


      cv::rectangle( image2,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );


    float depth = distance3D( correspondances->listSource[i].x ,correspondances->listSource[i].y , 0.0 ,   correspondances->listTarget[i].x , correspondances->listTarget[i].y , 0.0 );

      cv::rectangle( imageOutput,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( depth, depth, depth ),
                     -1,
                     8 );
  }


   cv::Mat fundMatCV( 3, 3,  CV_64FC1  );
   double fundMat[9]={0};
    std::vector<cv::Point2f> srcRANSACPoints;
    std::vector<cv::Point2f> dstRANSACPoints;

  if (useOpenCVEstimator)
  {
    fundMatCV = findFundamentalMat( srcPoints , dstPoints ,CV_FM_8POINT);
    fprintf(stderr,"Fundamental Matrix OpenCV: \n");
     std::cout << fundMatCV<<"\n";
  } else
  {
   fitHomographyTransformationMatchesRANSAC(
                                             1000 ,
                                             5.0 , 5.0 ,
                                             fundMat ,
                                             fundMatCV ,
                                             srcPoints ,
                                             dstPoints ,
                                             srcRANSACPoints ,
                                             dstRANSACPoints
                                            );

   fprintf(stderr,"Fundamental Matrix Mine : \n");
   for (unsigned int i=0; i<9; i+=3)
    { fprintf(stderr," %0.2f %0.2f %0.2f  \n", fundMat[i+0], fundMat[i+1], fundMat[i+2] ); }
  }




    cv::imwrite("rec1.jpg", image1);
    cv::imwrite("rec2.jpg", image2);
    cv::imwrite("recDepth.jpg", imageOutput);


  float camera1Mat[16]={0};
  snprintf(filename,512,"%s/%s1_camera.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * camera1 = readPointList( filename );
  camera1Mat[0] = camera1->listSource[0].x; camera1Mat[1] = camera1->listSource[0].y;
  camera1Mat[2] = camera1->listTarget[0].x; camera1Mat[3] = camera1->listTarget[0].y;

  camera1Mat[4] = camera1->listSource[1].x; camera1Mat[5] = camera1->listSource[1].y;
  camera1Mat[6] = camera1->listTarget[1].x; camera1Mat[7] = camera1->listTarget[1].y;

  camera1Mat[8] = camera1->listSource[2].x; camera1Mat[9] = camera1->listSource[2].y;
  camera1Mat[10]= camera1->listTarget[2].x; camera1Mat[11]= camera1->listTarget[2].y;

  fprintf(stderr,"Camera1 : \n");
  for (unsigned int i=0; i<14; i+=4)
   { fprintf(stderr," %0.2f %0.2f %0.2f %0.2f \n", camera1Mat[i+0], camera1Mat[i+1], camera1Mat[i+2], camera1Mat[i+3]); }


  float camera2Mat[16]={0};
  snprintf(filename,512,"%s/%s2_camera.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * camera2 = readPointList( filename );
  camera2Mat[0] = camera2->listSource[0].x; camera2Mat[1] = camera2->listSource[0].y;
  camera2Mat[2] = camera2->listTarget[0].x; camera2Mat[3] = camera2->listTarget[0].y;

  camera2Mat[4] = camera2->listSource[1].x; camera2Mat[5] = camera2->listSource[1].y;
  camera2Mat[6] = camera2->listTarget[1].x; camera2Mat[7] = camera2->listTarget[1].y;

  camera2Mat[8] = camera2->listSource[2].x; camera2Mat[9] = camera2->listSource[2].y;
  camera2Mat[10]= camera2->listTarget[2].x; camera2Mat[11]= camera2->listTarget[2].y;


  fprintf(stderr,"Camera2 : \n");
  for (unsigned int i=0; i<14; i+=4)
   { fprintf(stderr," %0.2f %0.2f %0.2f %0.2f \n", camera2Mat[i+0], camera2Mat[i+1], camera2Mat[i+2], camera2Mat[i+3]); }



/*
16.11 13.70 -67.35 -188.38
 0.83 -61.26 -27.99 -7.42
 0.17 -0.05 -0.08 0.57
*/



  return 1;
}




































