
/*
 * shm-client - client program to demonstrate shared memory.
 */
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

#include "SharedMemoryServer.h"

struct sharedMemoryDevices
{
    int shmid;
    key_t key;
    char *shm, *s;

    unsigned int activeFeeds;
    struct feedInformation feeds[16];
};

struct sharedMemoryDevices serverDevice[5];


int startSharedMemoryBroadcast(int devID, unsigned int width , unsigned int height )
{
  serverDevice[devID].key = 5678;
  serverDevice[devID].activeFeeds = 2;

  if ((serverDevice[devID].shmid = shmget(serverDevice[0].key, sizeof(struct sharedMemoryDevices) , IPC_CREAT |0666)) < 0) { fprintf(stderr,"Could not attach shared memory..!\n"); }
  if ((serverDevice[devID].feeds[0] = shmat(serverDevice[0].shmid, NULL, 0)) == (char *) -1) { perror("shmat"); }



}


int sharedMemoryServer_pushImageToRemote(int devID, int streamNumber , void* pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{

    /*
     * We need to get the segment named
     * "5678", created by the server.
     */

    /*
     * Locate the segment.
     */


    /*
     * Now we attach the segment to our data space.
     */



    exit(0);
}


int sharedMemoryTest()
{

}
