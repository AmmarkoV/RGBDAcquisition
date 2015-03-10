// DepthSenseGrabber
// http://github.com/ph4m

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include <sys/time.h>
#include <unistd.h>

#include <vector>
#include <exception>

#include "cv.h"
#include "highgui.h"

#include "DepthSenseGrabberOpenCV.hxx"
#include "../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx"
#include "../shared/ConversionTools.hxx"
#include "../shared/AcquisitionParameters.hxx"

int main(int argc, char* argv[])
{
    int flagExportType = FILETYPE_JPG; // FILETYPE_NONE, FILETYPE_JPG or FILETYPE_PNM

    int divideDepthBrightnessCV = 6;

    bool interpolateDepthFlag = 1;
    bool interpolateDepthAcqFlag = 0;
    bool interpolateColorFlag = 1;

    bool dispColorAcqFlag = 1;
    bool dispDepthAcqFlag = 1;
    bool dispColorSyncFlag = 1;
    bool dispDepthSyncFlag = 1;

    bool saveColorAcqFlag = 1;
    bool saveDepthAcqFlag = 0;
    bool saveColorSyncFlag = 0;
    bool saveDepthSyncFlag = 1;
    bool saveConfidenceFlag = 0;

    int flagColorFormat = FORMAT_VGA_ID; //QVGA, VGA, WXGA or NHD

    int widthColor, heightColor;
    switch (flagColorFormat) {
        case FORMAT_QVGA_ID:
            widthColor = FORMAT_QVGA_WIDTH;
            heightColor = FORMAT_QVGA_HEIGHT;
            break;
        case FORMAT_VGA_ID:
            widthColor = FORMAT_VGA_WIDTH;
            heightColor = FORMAT_VGA_HEIGHT;
            break;
        case FORMAT_WXGA_HEIGHT:
            widthColor = FORMAT_WXGA_WIDTH;
            heightColor = FORMAT_WXGA_HEIGHT;
            break;
        case FORMAT_NHD_HEIGHT:
            widthColor = FORMAT_NHD_WIDTH;
            heightColor = FORMAT_NHD_HEIGHT;
            break;
        default:
            printf("Unknown flagColorFormat");
            exit(EXIT_FAILURE);
    }

    int widthDepthAcq, heightDepthAcq;
    if (interpolateDepthFlag) {
        widthDepthAcq = FORMAT_VGA_WIDTH;
        heightDepthAcq = FORMAT_VGA_HEIGHT;
    } else {
        widthDepthAcq = FORMAT_QVGA_WIDTH;
        heightDepthAcq = FORMAT_QVGA_HEIGHT;
    }

    char fileNameColorAcq[50];
    char fileNameDepthAcq[50];
    char fileNameColorSync[50];
    char fileNameDepthSync[50];
    char fileNameConfidence[50];

    char baseNameColorAcq[20] = "colorFrame_0_";
    char baseNameDepthAcq[20] = "depthAcqFrame_0_";
    char baseNameColorSync[20] = "colorSyncFrame_0_";
    char baseNameDepthSync[20] = "depthFrame_0_";
    char baseNameConfidence[30] = "depthConfidenceFrame_0_";

    start_capture();

    uint16_t* pixelsDepthAcqQVGA = getPixelsDepthAcqQVGA();
    uint16_t* pixelsDepthAcqVGA = getPixelsDepthAcqVGA();
    uint8_t* pixelsColorAcq = getPixelsColorsAcq();
    uint16_t* pixelsDepthSync = getPixelsDepthSync();
    uint8_t* pixelsColorSyncQVGA = getPixelsColorSyncQVGA();
    uint8_t* pixelsColorSyncVGA = getPixelsColorSyncVGA();
    uint16_t* pixelsConfidenceQVGA = getPixelsConfidenceQVGA();

    uint16_t* pixelsDepthAcq;
    uint8_t* pixelsColorSync;
    if (interpolateDepthFlag) {
        pixelsDepthAcq = pixelsDepthAcqVGA;
        pixelsColorSync = pixelsColorSyncVGA;
    } else {
        pixelsDepthAcq = pixelsDepthAcqQVGA;
        pixelsColorSync = pixelsColorSyncQVGA;
    }




    ProjectionHelper* g_pProjHelper = NULL;
    StereoCameraParameters g_scp;



    IplImage *cv_depthAcqImage=NULL,
             *cv_colorAcqImage=NULL, // initialized in main, used in CBs
             *cv_depthSyncImage=NULL, // initialized in main, used in CBs
             *cv_colorSyncImage=NULL, // initialized in main, used in CBs
             *cv_emptyImage=NULL; // initialized in main, used in CBs
    CvSize cv_szDepthAcq=cvSize(widthDepthAcq,heightDepthAcq),
           cv_szColorAcq=cvSize(widthColor,heightColor);
    CvSize cv_szDepthSync = cv_szColorAcq, cv_szColorSync = cv_szDepthAcq;

    // VGA format color image
    cv_colorAcqImage=cvCreateImage(cv_szColorAcq,IPL_DEPTH_8U,3);
    if (cv_colorAcqImage==NULL)
    {
        printf("Unable to create color image buffer\n");
        exit(0);
    }

    // QVGA format depth image
    cv_depthAcqImage=cvCreateImage(cv_szDepthAcq,IPL_DEPTH_8U,1);
    if (cv_depthAcqImage==NULL)
    {
        printf("Unable to create depth image buffer\n");
        exit(0);
    }

    // QVGA format depth color image
    cv_depthSyncImage=cvCreateImage(cv_szDepthSync,IPL_DEPTH_8U,1);
    if (cv_depthSyncImage==NULL)
    {
        printf("Unable to create depth color image buffer\n");
        exit(0);
    }

    // QVGA format depth color image
    cv_colorSyncImage=cvCreateImage(cv_szColorSync,IPL_DEPTH_8U,3);
    if (cv_colorSyncImage==NULL)
    {
        printf("Unable to create color depth image buffer\n");
        exit(0);
    }

    // Empty image
    cv_emptyImage=cvCreateImage(cv_szColorSync,IPL_DEPTH_8U,1);
    if (cv_emptyImage==NULL)
    {
        printf("Unable to create empty image buffer\n");
        exit(0);
    }





    int frameCountPrevious = -1;
    while (true) {
        int frameCount = getFrameCount();
        int timeStamp = getTimeStamp();
        if (frameCount > frameCountPrevious) {
            frameCountPrevious = frameCount;
            printf("%d\n", frameCount);


            int countDepth = 0;
            for (int i=0; i<heightDepthAcq; i++) {
                for (int j=0; j<widthDepthAcq; j++) {
                   if (dispDepthAcqFlag || (saveDepthAcqFlag && (flagExportType == FILETYPE_JPG))) {
                       cvSet2D(cv_depthAcqImage,i,j,cvScalar(pixelsDepthAcq[countDepth]/divideDepthBrightnessCV));
                   }
                   //if (dispColorSyncFlag || (saveColorSyncFlag && (flagExportType == FILETYPE_JPG)))
                   //    cvSet2D(cv_colorSyncImage,i,j,cvScalar(pixelsColorSync[3*countDepth+2],pixelsColorSync[3*countDepth+1],pixelsColorSync[3*countDepth])); //BGR format
                   countDepth++;
                }
            }

            //if (dispColorAcqFlag) cvShowImage("Acq Color",cv_colorAcqImage);
            if (dispDepthAcqFlag) cvShowImage("Acq Depth",cv_depthAcqImage);
            //if (dispDepthSyncFlag) cvShowImage("Synchronized Depth",cv_depthSyncImage);
            //if (dispColorSyncFlag) cvShowImage("Synchronized Color",cv_colorSyncImage);
            //if (dispColorAcqFlag+dispColorSyncFlag+dispDepthAcqFlag+dispDepthSyncFlag == 0)  cvShowImage("Empty",cv_emptyImage);

            if (saveDepthAcqFlag) {
                sprintf(fileNameDepthAcq,"%s%05u.pnm",baseNameDepthAcq,frameCount);
                if (interpolateDepthFlag) saveDepthFramePNM(fileNameDepthAcq, pixelsDepthAcqVGA, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT, timeStamp);
                else saveDepthFramePNM(fileNameDepthAcq, pixelsDepthAcq, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
            if (saveColorAcqFlag) {
                sprintf(fileNameColorAcq,"%s%05u.pnm",baseNameColorAcq,frameCount);
                saveColorFramePNM(fileNameColorAcq, pixelsColorAcq, widthColor, heightColor, timeStamp);
            }
            if (saveDepthSyncFlag) {
                sprintf(fileNameDepthSync,"%s%05u.pnm",baseNameDepthSync,frameCount);
                saveDepthFramePNM(fileNameDepthSync, pixelsDepthSync, widthColor, heightColor, timeStamp);
            }
            if (saveColorSyncFlag) {
                sprintf(fileNameColorSync,"%s%05u.pnm",baseNameColorSync,frameCount);
                if (interpolateColorFlag) saveColorFramePNM(fileNameColorSync, pixelsColorSyncVGA, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT, timeStamp);
                else saveColorFramePNM(fileNameColorSync, pixelsColorSyncQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
            if (saveConfidenceFlag) {
                sprintf(fileNameConfidence,"%s%05u.pnm",baseNameConfidence,frameCount);
                saveDepthFramePNM(fileNameConfidence, pixelsConfidenceQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
            /*
            */

            char key = cvWaitKey(10);
            if (key==27)
            {
                printf("Quitting main loop from OpenCV\n");
                stop_capture();
            }


        }
    }


    return 0;
}
