#ifndef CALIBRATION_H_INCLUDED
#define CALIBRATION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

enum calibIntrinsics
{
  CALIB_INTR_FX = 0 ,
  CALIB_INTR_FY = 4 ,
  CALIB_INTR_CX = 2 ,
  CALIB_INTR_CY = 5
};


struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char intrinsicParametersSet;
  double intrinsic[9];
  double k1,k2,p1,p2,k3;

  /* CAMERA EXTRINSIC PARAMETERS */
  char extrinsicParametersSet;
  double extrinsicRotationRodriguez[3];
  double extrinsicTranslation[3];

  /*CAMERA DIMENSIONS ( WHEN RENDERING )*/
  double nearPlane,farPlane;
  unsigned int width;
  unsigned int height;

  double depthUnit;

  /*CONFIGURATION*/
  int imagesUsed;
  int boardWidth;
  int boardHeight;
  double squareSize;
};

float intAtof(char * str);

int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib);
int FocalLengthAndPixelSizeToCalibration(double focalLength , double pixelSize ,unsigned int width,unsigned int height, struct calibration * calib);
int ReadCalibration(char * filename,unsigned int width,unsigned int height,struct calibration * calib);
int WriteCalibration(char * filename,struct calibration * calib);


double * allocate4x4MatrixForPointTransformationBasedOnCalibration(struct calibration * calib);

int transform3DPointUsingCalibration(struct calibration * calib , float * x , float * y , float * z);

int transform2DProjectedPointTo3DPoint(struct calibration * calib , unsigned int x2d , unsigned int y2d  , unsigned short depthValue , float * x , float * y , float * z);

#ifdef __cplusplus
}
#endif


#endif
