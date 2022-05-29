/** @file OpenCLTools.h
 *  @brief  A collection of tools used for OpenCL stuff
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef OPENCL_TOOLS_H_INCLUDED
#define OPENCL_TOOLS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

unsigned long GetTickCountMicrosecondsOCL();

char * read_file(const char * filename,int *length );

const char *getErrorString(cl_int error);

int checkOpenCLError(int err,char * file , int  line);


int getBuildError(cl_program program,cl_device_id *devices);

#ifdef __cplusplus
}
#endif




#endif // OPENCL_TOOLS_H_INCLUDED
