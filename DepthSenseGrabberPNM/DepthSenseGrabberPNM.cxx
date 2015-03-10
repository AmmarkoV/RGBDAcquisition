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

//#include <DepthSense.hxx>

#include "DepthSenseGrabberPNM.hxx"
#include "../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx"
#include "../shared/ConversionTools.hxx"
#include "../shared/AcquisitionParameters.hxx"

int main(int argc, char* argv[])
{

    bool interpolateDepthFlag = 1;
    bool interpolateDepthAcqFlag = 0;
    bool interpolateColorFlag = 1;

    bool saveColorAcqFlag = 1;
    bool saveDepthAcqFlag = 0;
    bool saveColorSyncFlag = 0;
    bool saveDepthSyncFlag = 1;
    bool saveConfidenceFlag = 0;

    int flagColorFormat = 0; //0: VGA, 1: WXGA, 2: NHD

    int widthColor, heightColor;
    switch (flagColorFormat) {
        case 0:
            widthColor = FORMAT_VGA_WIDTH;
            heightColor = FORMAT_VGA_HEIGHT;
            break;
        case 1:
            widthColor = FORMAT_WXGA_WIDTH;
            heightColor = FORMAT_WXGA_HEIGHT;
            break;
        case 2:
            widthColor = FORMAT_NHD_WIDTH;
            heightColor = FORMAT_NHD_HEIGHT;
            break;
        default:
            printf("Unknown flagColorFormat");
            exit(EXIT_FAILURE);
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

    uint16_t* pixelsDepthAcq = getPixelsDepthAcq();
    uint16_t* pixelsDepthAcqVGA = getPixelsDepthAcqVGA();
    uint8_t* pixelsColorAcq = getPixelsColorsAcq();
    uint16_t* pixelsDepthSync = getPixelsDepthSync();
    uint8_t* pixelsColorSyncQVGA = getPixelsColorSyncQVGA();
    uint8_t* pixelsColorSyncVGA = getPixelsColorSyncVGA();
    uint16_t* pixelsConfidenceQVGA = getPixelsConfidenceQVGA();

    int frameCountPrevious = -1;
    while (true) {
        int frameCount = getFrameCount();
        int timeStamp = getTimeStamp();
        if (frameCount > frameCountPrevious) {
            frameCountPrevious = frameCount;
            printf("%d\n", frameCount);

            if (saveDepthAcqFlag) {
                sprintf(fileNameDepthAcq,"%s%05u.pnm",baseNameDepthAcq,frameCount);
                if (interpolateDepthFlag) saveRawDepthFrame(fileNameDepthAcq, pixelsDepthAcqVGA, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT, timeStamp);
                else saveRawDepthFrame(fileNameDepthAcq, pixelsDepthAcq, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
            if (saveColorAcqFlag) {
                sprintf(fileNameColorAcq,"%s%05u.pnm",baseNameColorAcq,frameCount);
                saveRawColorFrame(fileNameColorAcq, pixelsColorAcq, widthColor, heightColor, timeStamp);
            }
            if (saveDepthSyncFlag) {
                sprintf(fileNameDepthSync,"%s%05u.pnm",baseNameDepthSync,frameCount);
                saveRawDepthFrame(fileNameDepthSync, pixelsDepthSync, widthColor, heightColor, timeStamp);
            }
            if (saveColorSyncFlag) {
                sprintf(fileNameColorSync,"%s%05u.pnm",baseNameColorSync,frameCount);
                if (interpolateColorFlag) saveRawColorFrame(fileNameColorSync, pixelsColorSyncVGA, FORMAT_VGA_WIDTH, FORMAT_VGA_HEIGHT, timeStamp);
                else saveRawColorFrame(fileNameColorSync, pixelsColorSyncQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }
            if (saveConfidenceFlag) {
                sprintf(fileNameConfidence,"%s%05u.pnm",baseNameConfidence,frameCount);
                saveRawDepthFrame(fileNameConfidence, pixelsConfidenceQVGA, FORMAT_QVGA_WIDTH, FORMAT_QVGA_HEIGHT, timeStamp);
            }

        }
    }


    return 0;
}
