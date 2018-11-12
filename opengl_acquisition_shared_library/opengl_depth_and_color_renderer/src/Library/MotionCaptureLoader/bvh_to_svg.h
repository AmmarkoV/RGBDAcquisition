#ifndef BVH_TO_SVG_H_INCLUDED
#define BVH_TO_SVG_H_INCLUDED


#include "bvh_loader.h"
#include "bvh_transform.h"

int dumpBVHToSVG(
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
                 float * maximumObjectRotationValue
                 );

#endif // BVH_TO_SVG_H_INCLUDED
