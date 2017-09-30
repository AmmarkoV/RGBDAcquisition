
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

struct sharedMemoryDevices serverDevice[5];


int startSharedMemoryBroadcast()
{

}


int transmitSharedMemoryFrames(int devID)
{

    /*
     * We need to get the segment named
     * "5678", created by the server.
     */
    serverDevice[0].key = 5678;

    /*
     * Locate the segment.
     */
    if ((serverDevice[0].shmid = shmget(serverDevice[0].key, SHMSZ, 0666)) < 0) {
        perror("shmget");
    }

    /*
     * Now we attach the segment to our data space.
     */
    if ((serverDevice[0].shm = shmat(serverDevice[0].shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
    }

    /*
     * Now read what the server put in the memory.
     */
    for (serverDevice[0].s = serverDevice[0].shm; *serverDevice[0].s != NULL; serverDevice[0].s++)
        putchar(*serverDevice[0].s);
    putchar('\n');

    /*
     * Finally, change the first character of the
     * segment to '*', indicating we have read
     * the segment.
     */
    *serverDevice[0].shm = '*';

    exit(0);
}


int sharedMemoryTest()
{

}
