/** @file SharedMemory.h
 *  @brief A Way to Send and receive images via Linux SharedMemory
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef SHAREDMEMORY_H_INCLUDED
#define SHAREDMEMORY_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

int startSharedMemoryModule(unsigned int max_devs,char * settings);
int sharedMemoryTest();


struct feedInformation
{
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
};

#ifdef __cplusplus
}
#endif

#endif //SHAREDMEMORY_H_INCLUDED
