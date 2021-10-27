/** @file calibration.h
 *  @brief The main Acquisition library that handles plugins and provides .
 *
 *  This is a static library that gets linked to everything that needs calibration data ( and/or 3D Projected points )
 *  It is pretty compact and pretty much covers most of the things one would want it to do without having any external dependencies.
 *
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */



#ifndef CALIBRATION_H_INCLUDED
#define CALIBRATION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


/**
 * @brief A short hand  enumerator to make it easy to access the double intrinsic[9]; field of a struct calibration
 *
 * This enumerator holds all the possible values passed when and where moduleID is requested
 * These have an 1:1 relation to the modules that are supported and are used to reference them
 *
 */
enum calibIntrinsics
{
  CALIB_INTR_FX = 0 ,
  CALIB_INTR_FY = 4 ,
  CALIB_INTR_CX = 2 ,
  CALIB_INTR_CY = 5
};

/**
 * @brief The structure that holds the calibration data
 *
 *  The calibration structure covers everything from intrinsics / distortion  / dimensions to
 *  the extrinsic calibration of the input source , this gets set by each of the modules and is one of the
 *  core structures used in most things in RGBDAcquisition
 */
struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char intrinsicParametersSet;
  float intrinsic[9];
  float k1,k2,p1,p2,k3;

  /* CAMERA EXTRINSIC PARAMETERS */
  char extrinsicParametersSet;
  float extrinsicRotationRodriguez[3];
  float extrinsicTranslation[3];
  float extrinsic[16];

  /*CAMERA DIMENSIONS ( WHEN RENDERING )*/
  float nearPlane,farPlane;
  unsigned int width;
  unsigned int height;

  float depthUnit;

  /*CONFIGURATION*/
  int imagesUsed;
  int boardWidth;
  int boardHeight;
  float squareSize;
};


/**
 * @brief do an atof while forcing local to US
 * One of the problems encountered by people using this library is that calibration files would not get correctly parsed
 * depending on different conventions about floating point numbers from different countries
 * For example in france pi is written 3,14 and not 3.14 , that in turn made calibration files to be erroneously parsed
 * losing the fractional parts and becoming useless. internationalAtof enforces US locale so that we have a clear way to
 * save/load calibration files with no possible chances of failure between different systems with different locales.
 * @ingroup calibration
 * @param  string that will get parsed in to a float
 * @retval 1 if file Exists , 0 if file does not exist
 * @bug RGBDAcquisition forces floating points using US locale ( i.e. 3.14 instead of 3,14 ) Locales exist for a reason ,
 *      If someone wants to respect them he should consider setting RESPECT_LOCALES to 1 in calibration.c and recompiling everything
 */
float internationalAtof(const char * str);

/**
 * @brief  Populate a struct calibration with "default" values
 * @ingroup calibration
 * @param  width , With of frames
 * @param  height , Height of frames
 * @param  Pointer , Pointer to the calibration to be filled with the default values
 * @retval 1=Success , 0=Failure
 */
int NullCalibration(unsigned int width,unsigned int height, struct calibration * calib);

/**
 * @brief  Some Devices like PrimeSense devices provide Focal Length and Pixel Size as intrinsic calibration data , so we might want to use them
 *         to populate our structures
 * @ingroup calibration
 * @param  width , With of frames
 * @param  height , Height of frames
 * @param  Pointer , Pointer to the calibration to be filled with the default values
 * @retval 1=Success , 0=Failure
 */
int FocalLengthAndPixelSizeToCalibration(double focalLength , double pixelSize ,unsigned int width,unsigned int height, struct calibration * calib);



/**
 * @brief  Refresh a calibration structure with incremental data from calib file
 * @ingroup calibration
 * @param  filename , Path to file to read
 * @param  Pointer , Pointer to the calibration to be filled with the values of the file
 * @retval 1=Success , 0=Failure
 */
int RefreshCalibration(const char * filename,struct calibration * calib);


/**
 * @brief  Parse a new file and get the calibration specified into struct calibration * calib
 * @ingroup calibration
 * @param  filename , Path to file to read
 * @param  width , With of frames
 * @param  height , Height of frames
 * @param  Pointer , Pointer to the calibration to be filled with the values of the file
 * @retval 1=Success , 0=Failure
 */
int ReadCalibration(const char * filename,unsigned int width,unsigned int height,struct calibration * calib);

/**
 * @brief  Write a file and with the calibration specified into struct calibration * calib
 * @ingroup calibration
 * @param  filename , Path to file to read
 * @param  Pointer , Pointer to the calibration to be written to the file
 * @retval 1=Success , 0=Failure
 */
int WriteCalibration(const char * filename,struct calibration * calib);


int WriteCalibrationROS(const char * filename,struct calibration * calib);

/**
 * @brief  Write a basic calibration parameters to screen
 * @ingroup calibration
 * @param  Pointer , Pointer to the calibration to be printed on screen
 * @retval 1=Success , 0=Failure
 */
int PrintCalibration(struct calibration * calib);


/**
 * @brief  This call allocates a 4x4 Transformation matrix based on the extrinsics of the calibration file passed as a parameter
 *         Using this one can perform a massive amount of transformations without paying the overhead of constructing the matrix etc for every call
 *         In case of no extrinsics declared we return an identity transformation matrix
 * @ingroup calibration
 * @param  Pointer , Pointer to the calibration
 * @retval Pointer to a an array of 16 doubles (4x4) , 0=Failure
 */
float * allocate4x4MatrixForPointTransformationBasedOnCalibration(struct calibration * calib);



/**
 * @brief  This call will transform a 3D point from camera space to world coordinates using the extrinsics declared on the calibration
 * @ingroup calibration
 * @param  Pointer , Pointer to the calibration
 * @param  X3D , Input/Output of the 3D points , Input is camera Space , Output is world Space
 * @param  Y3D , Input/Output of the 3D points , Input is camera Space , Output is world Space
 * @param  Z3D , Input/Output of the 3D points , Input is camera Space , Output is world Space
 * @retval 1=Success , 0=Failure
 */
int transform3DPointUsingCalibration(struct calibration * calib , float * x , float * y , float * z);








/**
 * @brief  This call will transform a projected 2D point with a known depth value to a 3D point using a calibration
 * @ingroup calibration
 * @param  Pointer , Pointer to the calibration
 * @param  X2D , Input 2D point floating point subpixel accuracy
 * @param  Y2D , Input 2D point floating point subpixel accuracy
 * @param  DepthValue of the 2D point specified
 * @param  X3D , Output 3D point
 * @param  Y3D , Output 3D point
 * @param  Z3D , Output 3D point
 * @retval 1=Success , 0=Failure
 */
int transform2DFProjectedPointTo3DPoint(struct calibration * calib , float x2d , float y2d  , unsigned short depthValue , float * x , float * y , float * z);

/**
 * @brief  This call will transform a projected 2D point with a known depth value to a 3D point using a calibration
 * @ingroup calibration
 * @param  Pointer , Pointer to the calibration
 * @param  X2D , Input 2D point
 * @param  Y2D , Input 2D point
 * @param  DepthValue of the 2D point specified
 * @param  X3D , Output 3D point
 * @param  Y3D , Output 3D point
 * @param  Z3D , Output 3D point
 * @retval 1=Success , 0=Failure
 */
int transform2DProjectedPointTo3DPoint(struct calibration * calib , unsigned int x2d , unsigned int y2d  , unsigned short depthValue , float * x , float * y , float * z);







/**
 * @brief  This call will register a distorted color frame to a distorted depth map given the distortion models and the transformation between them ,
 * @ingroup calibration
 * @param  Pointer to distorted color image
 * @param  Width of color image
 * @param  Height of color image
 * @param  Calibration information of color image
 * @param  Pointer to distorted depth image
 * @param  Width of depth image
 * @param  Height of depth image
 * @param  Calibration information of depth image

 * @param  4x4 Matrix that transforms color to depth

 * @param  Output image width
 * @param  Output image height

 * @retval 0=Failure, Or a pointer to a registered color image
 */
unsigned char *  registerColorToDepthFrame(
                                           unsigned char * rgb , unsigned int rgbWidth , unsigned int rgbHeight , struct calibration * rgbCalibration ,
                                           unsigned short * depth , unsigned int depthWidth , unsigned int depthHeight , struct calibration * depthCalibration ,
                                           double * rotation3x3 , double * translation3x1 ,
                                           unsigned int * outputWidth , unsigned int * outputHeight
                                          );






/**
 * @brief  This call will register a distorted color frame to a distorted depth map given the distortion models and the transformation between them ,
 * @ingroup calibration
 * @param  Pointer to distorted color image
 * @param  Width of color image
 * @param  Height of color image
 * @param  Calibration information of color image
 * @param  Pointer to distorted depth image
 * @param  Width of depth image
 * @param  Height of depth image
 * @param  Calibration information of depth image

 * @param  4x4 Matrix that transforms color to depth

 * @param  Output image width
 * @param  Output image height

 * @retval 0=Failure, Or a pointer to a registered depth image
 */
unsigned short *  registerDepthToColorFrame(
                                           unsigned char * rgb , unsigned int rgbWidth , unsigned int rgbHeight , struct calibration * rgbCalibration ,
                                           unsigned short * depth , unsigned int depthWidth , unsigned int depthHeight , struct calibration * depthCalibration ,
                                           double * rotation3x3 , double * translation3x1 ,
                                           unsigned int * outputWidth , unsigned int * outputHeight
                                          );

#ifdef __cplusplus
}
#endif


#endif
