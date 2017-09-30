
/*
 * shm-client - client program to demonstrate shared memory.
 */
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

#define SHMSZ     27

struct sharedMemoryDevices
{
    int shmid;
    key_t key;
    char *shm, *s;
};

struct sharedMemoryDevices device[5];

int startSharedMemoryModule(unsigned int max_devs,char * settings)
{

}

int createSharedMemoryDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  /*
   * We need to get the segment named * "5678", created by the server.
   */
  device[devID].key = 5678;

  /* * Locate the segment. */
  if ((device[devID].shmid = shmget(device[devID].key, SHMSZ, 0666)) < 0) { return 0; }

  /* * Now we attach the segment to our data space. */
  if ((device[devID].shm = shmat(device[devID].shmid, NULL, 0)) == (char *) -1) { return 0; }

}

int snapSharedMemoryFrames(int devID)
{

    /*
     * Now read what the server put in the memory.
     */
    for (device[devID].s = device[devID].shm; *device[devID].s != NULL; device[devID].s++)
        putchar(*device[devID].s);
    putchar('\n');

    /*
     * Finally, change the first character of the
     * segment to '*', indicating we have read
     * the segment.
     */
    *device[devID].shm = '*';

}

