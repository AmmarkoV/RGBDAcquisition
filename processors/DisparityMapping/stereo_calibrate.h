#ifndef STEREO_CALIBRATE_H_INCLUDED
#define STEREO_CALIBRATE_H_INCLUDED

int doCalibrationStep(cv::Mat * leftImgRGB ,
                      cv::Mat * rightImgRGB ,
                        cv::Mat * leftImgGray ,
                        cv::Mat * rightImgGray,
                      unsigned int horizontalSquares,unsigned int verticalSquares,float calibSquareSize,
                      char * disparityCalibrationOutputPath);


int finalizeCalibration(char * outputFolder,int nx, int ny, float _squareSize);

#endif // STEREO_CALIBRATE_H_INCLUDED
