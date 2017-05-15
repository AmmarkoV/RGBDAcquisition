#ifndef JSONCOCOSKELETON_H_INCLUDED
#define JSONCOCOSKELETON_H_INCLUDED


#if REDIFINE_COCO_SKELETON
static const char * COCOBodyNames[] =
{
  "Nose",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "RHip",
  "RKnee",
  "RAnkle",
  "LHip",
  "LKnee",
  "LAnkle",
  "REye",
  "LEye",
  "REar",
  "LEar",
  "Bkg",
//=================
    "End of Joint Names"
};


enum COCOSkeletonJoints
{
  COCO_Nose,
  COCO_Neck,
  COCO_RShoulder,
  COCO_RElbow,
  COCO_RWrist,
  COCO_LShoulder,
  COCO_LElbow,
  COCO_LWrist,
  COCO_RHip,
  COCO_RKnee,
  COCO_RAnkle,
  COCO_LHip,
  COCO_LKnee,
  COCO_LAnkle,
  COCO_REye,
  COCO_LEye,
  COCO_REar,
  COCO_LEar,
  COCO_Bkg,
   //---------------------
  COCO_PARTS
};
#endif // REDIFINE_COCO_SKELETON


int parseJsonCOCOSkeleton(const char * filename , struct skeletonCOCO * skel);

#endif // JSONCOCOSKELETON_H_INCLUDED
