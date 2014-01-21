#include "FaceDetection.h"
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <cxcore.h>

IplImage  *image=0;
char * opencv_pointer_retainer=0; // This is a kind of an ugly hack ( see lines noted with UGLY HACK ) to minimize memcpying between my VisCortex and OpenCV , without disturbing OpenCV

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;


void InitFaceDetection(char * haarCascadePath , unsigned int x,unsigned int y)
{
    char filename[512];
    strcpy (filename,haarCascadePath );

    /* load the classifier
       note that I put the file in the same directory with
       this code */
    cascade = ( CvHaarClassifierCascade* ) cvLoad( filename, 0, 0, 0 );

    /* setup memory buffer; needed by the face detector */
    storage = cvCreateMemStorage( 0 );

    image = cvCreateImage( cvSize(x,y), IPL_DEPTH_8U, 3 );
    opencv_pointer_retainer = image->imageData; // UGLY HACK
}

void CloseFaceDetection()
{
    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

    image->imageData = opencv_pointer_retainer; // UGLY HACK
    cvReleaseImage( &image );
}


unsigned int DetectFaces(unsigned char * colorPixels)
{
    //fprintf(stderr,"Detecting Faces\n");
    if  (colorPixels == 0 )  { return 0; }

    /* detect faces */
    image->imageData=(char*) colorPixels; // UGLY HACK


    CvSeq *faces = cvHaarDetectObjects
           (
            image,
            cascade,
            storage,
            1.1,
            3,
            0 /*CV_HAAR_DO_CANNY_PRUNNING*/
            , cvSize( 40, 40 )
            , cvSize( 110, 142) // <--- This might have to be commented out if compiled with C++ :P
            );

    /* for each face found, draw a red box */
    int i;
    for( i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ )
    {
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );


        /*
        if ( settings[REMEMBER_FACES] )
         {
           char timestamped_filename[512]={0};
           timestamped_filename[0]=0; timestamped_filename[1]=0;
           GetANewSnapShotFileName(timestamped_filename,"memfs/faces/face_snap",".ppm");
           SaveRegisterPartToFile(timestamped_filename,vid_reg, r->x , r->y , r->width , r->height );
         }
        */

        fprintf(stderr,"Face%u @ %u , %u ( size %u,%u ) \n",i,r->x , r->y ,  r->width , r->height );
       /* AddToFeatureList(  video_register[vid_reg].faces  ,
                           r->x , r->y , 0 ,
                           r->width , r->height , 0
                         );*/

    }

	return 0;
}

