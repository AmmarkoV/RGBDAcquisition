#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"

int dumpBVHToSVGCSV(
                 const char * directory ,
                 int convertToSVG,
                 int convertToCSV,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height,

                 unsigned int useOriginalPositionsRotations,
                 float * cameraPositionOffset,
                 float * cameraRotationOffset,
                 float * objectRotationOffset,

                 unsigned int randomizePoses,
                 float * minimumObjectPositionValue,
                 float * maximumObjectPositionValue,
                 float * minimumObjectRotationValue,
                 float * maximumObjectRotationValue,

                 unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                 unsigned int encodeRotationsAsRadians
                 );


#endif // BVH_EXPORT_H_INCLUDED
