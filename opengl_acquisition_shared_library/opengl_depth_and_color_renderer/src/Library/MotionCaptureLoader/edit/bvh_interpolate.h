#ifndef BVH_INTERPOLATE_H_INCLUDED
#define BVH_INTERPOLATE_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Interpolates motion in the BVH motion capture data.
 *
 * This function interpolates the motion in the BVH motion capture data by adding interpolated frames between each existing frame. It doubles the number of frames and updates the motion values accordingly. The interpolation is done by averaging the motion values of adjacent frames.
 *
 * @param mc Pointer to the BVH_MotionCapture structure containing motion capture data.
 *
 * @return Returns 1 if the function executed successfully, or 0 if there was an error or if the BVH_MotionCapture pointer is null.
 */
int bvh_InterpolateMotion(struct BVH_MotionCapture * mc);

#ifdef __cplusplus
}
#endif

#endif // BVH_INTERPOLATE_H_INCLUDED
