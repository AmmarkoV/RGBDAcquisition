#ifndef BVH_RENAME_H_INCLUDED
#define BVH_RENAME_H_INCLUDED


#include "../bvh_loader.h"


void lowercase(char *a);

/**
* @brief Different motion capture systems produce different types of joint names. For example lhip can be named lefthip,leftupleg,lthigh,leftupperLeg etc.
*        This call renames them in order to ensure better compatibility with various different motion capture files..
* @ingroup BVH
* @param  BVH Structure
*/
void bvh_renameJointsForCompatibility(struct BVH_MotionCapture * bvhMotion);

#endif
