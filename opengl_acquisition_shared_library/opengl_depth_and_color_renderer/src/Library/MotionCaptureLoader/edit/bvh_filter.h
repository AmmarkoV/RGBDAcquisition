#ifndef BVH_FILTER_H_INCLUDED
#define BVH_FILTER_H_INCLUDED


#include "../bvh_loader.h"

#include "../export/bvh_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Filters out poses that are gimbal locked based on the specified threshold.
 *
 * This function filters out poses from the BVH motion capture data that are gimbal locked based on the specified threshold. It checks each motion value in each frame and determines if it falls within the gimbal lock range. Frames that have motion values within the gimbal lock range are considered to be gimbal locked and are marked for hiding.
 *
 * @param mc Pointer to the BVH_MotionCapture structure containing motion capture data.
 * @param threshold The threshold value used to determine gimbal lock range.
 *
 * @return Returns 1 if the function executed successfully, or 0 if there was an error.
 */
int filterOutPosesThatAreGimbalLocked(struct BVH_MotionCapture * mc,float threshold);

/**
 * @brief Filters out poses that are close to specified rules based on command line arguments.
 *
 * This function filters out poses from the BVH motion capture data that are close to the specified rules based on the provided command line arguments. It calculates the distance between specified joints in each frame and checks if the distances fall within the specified range for each rule. Frames that match all the rules are considered to be a match and are marked for hiding.
 *
 * @param mc Pointer to the BVH_MotionCapture structure containing motion capture data.
 * @param argc The number of command line arguments.
 * @param argv An array of strings containing the command line arguments.
 *
 * @return Returns 1 if the function executed successfully, or 0 if there was an error.
 */
int filterOutPosesThatAreCloseToRules(struct BVH_MotionCapture * bvhMotion,int argc,const char **argv);

/**
 * @brief Probes for filter rules based on command line arguments and performs rule-based filtering on BVH motion capture frames.
 *
 * This function probes for filter rules based on the provided command line arguments and applies rule-based filtering on the frames of the BVH motion capture data. It calculates the distance between specified joints in each frame and checks if the distances fall within the specified range for each rule. Frames that match all the rules are considered to be a match and are marked for hiding.
 *
 * @param mc Pointer to the BVH_MotionCapture structure containing motion capture data.
 * @param argc The number of command line arguments.
 * @param argv An array of strings containing the command line arguments.
 *
 * @return Returns 1 if the function executed successfully, or 0 if there was an error.
 */
int probeForFilterRules(struct BVH_MotionCapture * mc,int argc,const char **argv);

#ifdef __cplusplus
}
#endif

#endif
