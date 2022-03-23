/** @file bvhLibrary.h
 *  @brief  BVH file parser part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            This is the central header for the BVH library in order to compile it not as an executable file but as a real library!
            To enable building as a library please compile the code with -DBVH_USE_AS_A_LIBRARY so that there is no main function included!
            Don't forget, to check generated symbols : nm -gD libBVHConverter.so
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef BVH_STANDALONE_LIBRARY_H_INCLUDED
#define BVH_STANDALONE_LIBRARY_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


int bvhConverter_loadAtomic(const char *path);


int bvhConverter_rendererConfigurationAtomic(const char ** labels,const float * values,int numberOfElements);
int bvhConverter_processFrame(int frameID);
int bvhConverter_getJointNameJointID(const char * jointName);

const char * bvhConverter_getJointNameFromJointID(int jointID);

int bvhConverter_getJointParent(int jointID);

float  bvhConverter_get3DX(int jointID);
float  bvhConverter_get3DY(int jointID);
float  bvhConverter_get3DZ(int jointID);

float  bvhConverter_get2DX(int jointID);
float  bvhConverter_get2DY(int jointID);

int bvhConverter_modifyAtomic(const char ** labels,const float * values,int numberOfElements,int frameID);

int bvhConverter(int argc,const char **argv);


#ifdef __cplusplus
}
#endif

#endif // BVH_STANDALONE_LIBRARY_H_INCLUDED
