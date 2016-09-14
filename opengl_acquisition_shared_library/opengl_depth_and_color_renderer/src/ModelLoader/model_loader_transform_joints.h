#ifndef MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
#define MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED


struct TRI_Transform
{
  unsigned int boneParent;
  double finalTransform[16];
};

#endif // MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
