#ifndef CALIBRATION_H_INCLUDED
#define CALIBRATION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


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

  /*CAMERA DIMENSIONS*/
  double nearPlane,farPlane;
  unsigned int width;
  unsigned int height;

  /*CONFIGURATION*/
  int imagesUsed;
  int boardWidth;
  int boardHeight;
  double squareSize;
};

int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib);
int ReadCalibration(char * filename,unsigned int width,unsigned int height,struct calibration * calib);
int WriteCalibration(char * filename,struct calibration * calib);

#ifdef __cplusplus
}
#endif


#endif
