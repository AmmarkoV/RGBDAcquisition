#include "reconstruction.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        // printf("A) vals are |%s|%s|=======| \n", num1,num2 );
        char * num3 = strchr(num2 , ' ');
        while (*num3==' ') { *num3=0; ++num3; }
        // printf("B) vals are |%s|%s|=======| \n", num1,num2 );
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









int  reconstruct3D(const char * filenameLeft )
{
  char filename[512]={0};
  snprintf(filename,512,"%s/%s_matches.txt",filenameLeft ,filenameLeft);

  struct Point2DCorrespondance * correspondances = readPointList( filename );
  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {

      printf("#%u (%0.2f , %0.2f ) -> ( %0.2f , %0.2f) \n" , i ,
        correspondances->listSource[i].x ,
        correspondances->listSource[i].y ,
        correspondances->listTarget[i].x ,
        correspondances->listTarget[i].y  );
  }

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



  snprintf(filename,512,"%s/%s1.jpg",filenameLeft ,filenameLeft);
  cv::Mat image1 = cv::imread(filename  , CV_LOAD_IMAGE_COLOR);
  if(! image1.data ) { fprintf(stderr,"Image1 missing \n"); return 0; }

  snprintf(filename,512,"%s/%s2.jpg",filenameLeft ,filenameLeft);
  cv::Mat image2 = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if(! image2.data ) { fprintf(stderr,"Image2 missing \n"); return 0; }


  return 1;
}




































