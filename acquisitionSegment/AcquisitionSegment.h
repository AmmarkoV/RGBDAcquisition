/** @file AcquisitionSegment.h
 *  @brief This is a small library that gets color/depth frames and based on rules set on
 *  struct SegmentationFeaturesRGB and  struct SegmentationFeaturesDepth segments the input pictures
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug There needs to be a better way to describe point lists to make the various filters more variable
 */

#ifndef ACQUISITIONSEGMENT_H_INCLUDED
#define ACQUISITIONSEGMENT_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif



#include "../tools/Calibration/calibration.h"


/**
 * @brief An enumerator of the ways we can combine the Color with the Depth Streams
 *  DONT_COMBINE will just do the operations specified in struct SegmentationFeaturesRGB to the RGB frame , and the operations specified int struct SegmentationFeaturesDepth to the Depth frame without combining them.
 *  COMBINE_AND will apply changes both present on RGB and Depth Frames to both of their outputs
 *  COMBINE_OR  will apply changes both present on either RGB or Depth Frames to both of their outputs
 *  COMBINE_XOR  will apply changes that are only present on either RGB or Depth Frames to both of their outputs , if they both have the change it wont be applied (not very useful)
 *  COMBINE_KEEP_ONLY_RGB  will force changes on RGB frame to be applied to the Depth frame as well
 *  COMBINE_KEEP_ONLY_DEPTH  will force changes on Depth frame to be applied to the RGB frame as well
 */
enum combinationModesEnumerator
{
  DONT_COMBINE=0,
  COMBINE_AND   ,
  COMBINE_OR    ,
  COMBINE_XOR   ,
  COMBINE_KEEP_ONLY_RGB  ,
  COMBINE_KEEP_ONLY_DEPTH ,
  COMBINE_SWAP ,
  COMBINE_RGBFULL_DEPTH_USE_RGB,
  COMBINE_DEPTHFULL_RGB_USE_DEPTH,
  //-----------------------------
  NUMBER_OF_COMBINATION_MODES
};

/**
 * @brief A structure that holds the points to be erased using flood fill
 * @bug This should be integrated into a primitive point list declared somewhere else for the whole library to use
 */
struct floodErasePoints
{
  int totalPoints ;
  int source;
  int target;
  unsigned int pX[32];
  unsigned int pY[32];
  unsigned int threshold[32];
};


/**
 * @brief The structure that holds the cirteria to do RGB Segmentation
 */
struct SegmentationFeaturesRGB
{
   unsigned int saveRGB;
   unsigned char eraseColorR , eraseColorG , eraseColorB;

   unsigned int minR ,  minG ,  minB;
   unsigned int maxR , maxG , maxB;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

   unsigned char replaceR , replaceG , replaceB;
   char enableReplacingColors;


   unsigned char enableRGBMotionDetection;
   unsigned char * firstRGBFrame;
   unsigned int firstRGBFrameByteSize;
   unsigned int motionRThreshold,motionGThreshold,motionBThreshold;

   struct floodErasePoints floodErase;

   //----------------
   int isInitialized;
   int invert;
};


/**
 * @brief The structure that holds the cirteria to do Depth Segmentation
 */
struct SegmentationFeaturesDepth
{
   unsigned int saveDepth;
   unsigned int minDepth, maxDepth;

   unsigned int minX , maxX;
   unsigned int minY , maxY;

   struct floodErasePoints floodErase;

   unsigned char enableBBox;
   double bboxX1,bboxY1,bboxZ1,bboxX2,bboxY2,bboxZ2;

   unsigned char enableDepthMotionDetection;
   unsigned short * firstDepthFrame;
   unsigned int firstDepthFrameByteSize;
   unsigned int motionDistanceThreshold;


   unsigned char autoPlaneSegmentation;
   unsigned char enablePlaneSegmentation;
   unsigned char doNotGenerateNormalFrom3Points;
   double p1[3];
   double p2[3];
   double p3[3];

   double center[3];
   double normal[3];
   unsigned int autoPlaneSegmentationMinimumDistancePoint;
   unsigned int autoPlaneSegmentationMaximumDistancePoint;
   double planeNormalOffset;
   double planeNormalSize;


   //----------------
   int isInitialized;
   int invert;
};



unsigned char * splitStereo(unsigned char * source ,
                            unsigned int feed,
                            unsigned int * width ,
                            unsigned int * height
                           );


/**
 * @brief  Allocate an array with width*height size that describes which pixels we our configuration keeps( set as ones ) and it discards ( set as zeros )
 * @ingroup acquisitionSegment
 * @param Pointer to RGB Input
 * @param Width of RGB Input
 * @param Height of RGB Input
 * @param Structure holding the segmentation criteria for RGB Frame
 * @param Calibration of the camera that produced the RGB Input
 * @retval Pointer to a width*height array with which pixels to keep and which to discard , should be freed when user is done with it , 0 = Failure
 */
unsigned char * segmentRGBFrame(unsigned char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf, struct calibration * calib,unsigned int * selectedPixels);



/**
 * @brief  Allocate an array with width*height size that describes which pixels we our configuration keeps( set as ones ) and it discards ( set as zeros )
 * @ingroup acquisitionSegment
 * @param Pointer to Depth Input
 * @param Width of Depth Input
 * @param Height of Depth Input
 * @param Structure holding the segmentation criteria for Depth Frame
 * @param Calibration of the camera that produced the Depth Input
 * @retval Pointer to a width*height array with which pixels to keep and which to discard , should be freed when user is done with it , 0 = Failure
 */
unsigned char * segmentDepthFrame(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf, struct calibration * calib,unsigned int * selectedPixels);




/**
 * @brief  Segment RGB/Depth frames according to our Configuration and Calibration
 * @ingroup acquisitionSegment
 * @param Pointer to RGB Input
 * @param Pointer to Depth Input
 * @param Width of Depth Input
 * @param Height of Depth Input
 * @param Structure holding the segmentation criteria for RGB Frame
 * @param Structure holding the segmentation criteria for Depth Frame
 * @param Calibration of the camera that produced the Depth Input
 * @param  Combination mode
 * @retval 1 = Success , 0 = Failure
 */
int   segmentRGBAndDepthFrame (    unsigned char * RGB ,
                                   unsigned short * Depth ,
                                   unsigned int width , unsigned int height ,
                                   struct SegmentationFeaturesRGB * segConfRGB ,
                                   struct SegmentationFeaturesDepth * segConfDepth,
                                   struct calibration * calib ,
                                   int combinationMode
                               );



/**
 * @brief  Initialize an empty segConfRGB with default values
 * @ingroup acquisitionSegment
 * @param Pointer to the RGB Configuration structure
 * @param Width of RGB Frames for Input
 * @param Height of RGB Frames for Input
 * @retval 1 = Success , 0 = Failure
 */
int initializeRGBSegmentationConfiguration(struct SegmentationFeaturesRGB * segConfRGB , unsigned int width , unsigned int height );


/**
 * @brief  Initialize an empty segConfDepth with default values
 * @ingroup acquisitionSegment
 * @param Pointer to the Depth Configuration structure
 * @param Width of Depth Frames for Input
 * @param Height of Depth Frames for Input
 * @retval 1 = Success , 0 = Failure
 */
int initializeDepthSegmentationConfiguration(struct SegmentationFeaturesDepth* segConfDepth , unsigned int width , unsigned int height );


/**
 * @brief Copy an RGB Segmentation to another RGB Segmentation
 * @ingroup acquisitionSegment
 * @param Pointer to target destination RGB Segmentation
 * @param Pointer to source RGB Segmentation
 * @retval 1 = Success , 0 = Failure
 */
int copyRGBSegmentation(struct SegmentationFeaturesRGB* target, struct SegmentationFeaturesRGB* source);


/**
 * @brief Copy a Depth Segmentation to another Depth Segmentation
 * @ingroup acquisitionSegment
 * @param Pointer to target destination Depth Segmentation
 * @param Pointer to source Depth Segmentation
 * @retval 1 = Success , 0 = Failure
 */
int copyDepthSegmentation(struct SegmentationFeaturesDepth* target, struct SegmentationFeaturesDepth* source);


/**
 * @brief  Print depth segmentation to stderr
 * @ingroup acquisitionSegment
 * @param A string with a comment ( on what has been printed )
 * @param Depth Configuration we want to print out
 * @retval 1 = Success , 0 = Failure
 */
int printDepthSegmentationData(char * label , struct SegmentationFeaturesDepth * dat);



/**
 * @brief  getDepthBlob for Depth Frame block starting at sX,sY  and with size width/height
 * @ingroup acquisitionSegment
 * @param Depth Frame
 * @param Width of Depth Frame
 * @param Height of Depth Frame
 * @param Block for DepthBlob Extraction Start X
 * @param Block for DepthBlob Extraction Start Y
 * @param Block for DepthBlob Extraction Width
 * @param Block for DepthBlob Extraction Height
 * @param Output Depth Blob 3d Center X
 * @param Output Depth Blob 3d Center Y
 * @param Output Depth Blob 3d Center Z
 * @retval 1 = Success , 0 = Failure
 */
int segmentGetDepthBlobAverage(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                               unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                               float * centerX , float * centerY , float * centerZ);




/**
 * @brief  get dimensions of a depth blob
 * @ingroup acquisitionSegment
 * @param Depth Frame
 * @param Width of Depth Frame
 * @param Height of Depth Frame
 * @param Block for DepthBlob Extraction Start X
 * @param Block for DepthBlob Extraction Start Y
 * @param Block for DepthBlob Extraction Width
 * @param Block for DepthBlob Extraction Height
 * @param Output Depth Blob 3d Size X
 * @param Output Depth Blob 3d Size Y
 * @param Output Depth Blob 3d Size Z
 * @retval 1 = Success , 0 = Failure
 */
int segmentGetDepthBlobDimensions(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                                  unsigned int sX,unsigned int sY,unsigned int width,unsigned int height,
                                  float * centerX , float * centerY , float * centerZ);





int segmentGetPointArea(unsigned short * frame , unsigned int frameWidth , unsigned int frameHeight,
                        unsigned int *sX,unsigned int *sY,unsigned int *width,unsigned int *height );


/**
 * @brief  stub , planned functionality for the future
 * @ingroup acquisitionSegment
 */
unsigned char * mallocSelectVolume(unsigned short * depthFrame , unsigned int frameWidth , unsigned int frameHeight ,
                                   unsigned int sX,unsigned int sY , float sensitivity );


/**
 * @brief  Save RGB/Depth Segmentation and the combination mode to be used as arguments that can be supplied to a program
 * @ingroup acquisitionSegment
 * @param Filename of output file
 * @param Pointer to RGB Segmentation criteria
 * @param Pointer to Depth Segmentation criteria
 * @param Pointer to the combination method
 * @retval 1 = Success , 0 = Failure
 */
int saveSegmentationDataToFile(char* filename , struct SegmentationFeaturesRGB * rgbSeg , struct SegmentationFeaturesDepth * depthSeg , unsigned int combinationMode);

/**
 * @brief  Parse command line arguments to populate RGB/Depth Segmentation and the combination mode to be used
 * @ingroup acquisitionSegment
 * @param Number of arguments , argc from int main(int argc , char * argv[])
 * @param Pointer to list of arguments , argv from int main(int argc , char * argv[])
 * @param Pointer to RGB Segmentation criteria to be populated after parsing the arguments
 * @param Pointer to Depth Segmentation criteria to be populated after parsing the arguments
 * @param Pointer to the combination method to be populated after parsing the arguments
 * @retval 1 = Success , 0 = Failure
 */
int loadSegmentationDataFromArgs(int argc, char *argv[] , struct SegmentationFeaturesRGB * rgbSeg , struct SegmentationFeaturesDepth * depthSeg , unsigned int * combinationMode);
#ifdef __cplusplus
}
#endif

#endif // ACQUISITIONSEGMENT_H_INCLUDED
