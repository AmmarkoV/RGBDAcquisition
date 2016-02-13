#ifndef MODULES_H_INCLUDED
#define MODULES_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif
 

enum MODULE_CAPABILITIES
{
  CAP_VERSION=0,
  CAP_LIVESTREAM,
  CAP_PROVIDES_LOCATIONS,

  //---------------------------
  CAP_ENUM_LIST_VERSION
};


#ifdef __cplusplus
}
#endif

#endif // MODULES_H_INCLUDED

