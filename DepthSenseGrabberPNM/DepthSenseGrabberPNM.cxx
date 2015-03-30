// DepthSenseGrabber
// http://github.com/ph4m

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//#include <sys/time.h>
//#include <unistd.h>

#include <vector>
#include <exception>

#include "DepthSenseGrabberPNM.hxx"
#include "../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx"
#include "../shared/ConversionTools.hxx"
#include "../shared/AcquisitionParameters.hxx"

int main(int argc, char* argv[])
{

    bool interpolateDepthFlag = 1;

    bool saveColorAcqFlag   = 1;
    bool saveDepthAcqFlag   = 1;
    bool saveColorSyncFlag  = 1;
    bool saveDepthSyncFlag  = 1;
    bool saveConfidenceFlag = 1;

    bool buildColorSyncFlag = saveColorSyncFlag;
    bool buildDepthSyncFlag = saveDepthSyncFlag;
    bool buildConfidenceFlag = saveConfidenceFlag;

    int flagColorFormat = FORMAT_VGA_ID; // VGA, WXGA or NHD

    int widthColor, heightColor;
    switch (flagColorFormat) {
        case FORMAT_VGA_ID:
            widthColor = FORMAT_VGA_WIDTH;
            heightColor = FORMAT_VGA_HEIGHT;
            break;
        case FORMAT_WXGA_ID:
            widthColor = FORMAT_WXGA_WIDTH;
            heightColor = FORMAT_WXGA_HEIGHT;
            break;
        case FORMAT_NHD_ID:
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

    start_capture(flagColorFormat,
                  interpolateDepthFlag,
                  buildColorSyncFlag, buildDepthSyncFlag, buildConfidenceFlag);

    uint16_t* pixelsDepthAcq;
    uint8_t* pixelsColorSync;
    uint8_t* pixelsColorAcq = getPixelsColorsAcq();
    uint16_t* pixelsDepthSync = getPixelsDepthSync();
    uint16_t* pixelsConfidenceQVGA = getPixelsConfidenceQVGA();
    if (interpolateDepthFlag) {
        pixelsDepthAcq = getPixelsDepthAcqVGA();
        pixelsColorSync = getPixelsColorSyncVGA();
    } else {
        pixelsDepthAcq = getPixelsDepthAcqQVGA();
        pixelsColorSync = getPixelsColorSyncQVGA();
    }

    int frameCountPrevious = -1;
    while (true) {
        int frameCount = getFrameCount();
        int timeStamp = getTimeStamp();
        if (frameCount > frameCountPrevious) {
            frameCountPrevious = frameCount;
            printf("%d\n", frameCount);

            if (saveDepthAcqFlag) {
                sprintf(fileNameDepthAcq,"%s%05u.pnm",baseNameDepthAcq,frameCount);
                saveDepthFramePNM(fileNameDepthAcq, pixelsDepthAcq, widthDepthAcq, heightDepthAcq, timeStamp);
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
                saveColorFramePNM(fileNameColorSync, pixelsColorSync, widthDepthAcq, heightDepthAcq, timeStamp);
            }
            if (saveConfidenceFlag) {
                sprintf(fileNameConfidence,"%s%05u.pnm",baseNameConfidence,frameCount);
                saveDepthFramePNM(fileNameConfidence, pixelsConfidenceQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
        }
    }


    return 0;
}
