#include "FaceDetection.h"
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <cxcore.h>

IplImage  *image=0;
char * opencv_pointer_retainer=0; // This is a kind of an ugly hack ( see lines noted with UGLY HACK ) to minimize memcpying between my VisCortex and OpenCV , without disturbing OpenCV

CvHaarClassifierCascade *cascade=0;
CvMemStorage            *storage=0;


int InitFaceDetection(char * haarCascadePath , unsigned int width ,unsigned int height)
{
    /* load the classifier
       note that I put the file in the same directory with
       this code */
    cascade = ( CvHaarClassifierCascade* ) cvLoad(haarCascadePath, 0, 0, 0 );
    if (cascade==0) { fprintf(stderr,"Could not load cascade file %s\n",haarCascadePath); return 0; }
    /* setup memory buffer; needed by the face detector */
    storage = cvCreateMemStorage( 0 );
    if (storage==0) { fprintf(stderr,"Could not allocate storage \n"); return 0; }

    image = cvCreateImage( cvSize(width,height), IPL_DEPTH_8U, 3 );
    if (image==0) { fprintf(stderr,"Could not allocate image \n"); return 0; }

    opencv_pointer_retainer = image->imageData; // UGLY HACK
    return 1;
}

int CloseFaceDetection()
{
    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

    image->imageData = opencv_pointer_retainer; // UGLY HACK
    cvReleaseImage( &image );
}


unsigned int DetectFaces(unsigned char * colorPixels , unsigned int maxHeadSize,unsigned int minHeadSize)
{
    if  (colorPixels == 0 )  { return 0; }
    if (cascade==0)  { return 0; }
    if (storage==0)  { return 0; }
    if (image==0)    { return 0; }

    /* detect faces */
    image->imageData=(char*) colorPixels; // UGLY HACK

    unsigned int maxX=(unsigned int) (minHeadSize)       , maxY=minHeadSize;
    unsigned int minX=(unsigned int) (0.77*maxHeadSize)  , minY=maxHeadSize;
    //fprintf(stderr,"Detect Faces Min %u,%u Max %u,%u \n",minX,minY,maxX,maxY);

    CvSeq *faces = cvHaarDetectObjects
           (
            image,
            cascade,
            storage,
            1.1,
            3,
            0 /*CV_HAAR_DO_CANNY_PRUNNING*/
            , cvSize(minX,minY)
            , cvSize(maxX,maxY) // <--- This might have to be commented out if compiled with C++ :P
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

	return 1;
}

