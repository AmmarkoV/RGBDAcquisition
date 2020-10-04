/** @file example.c
 *  @brief  An example using pthreadWorkerPool.h to organize a large number of concurrently working threads without
 *  too many lines of code or difficulty understanding what is happening
 *  https://github.com/AmmarkoV/PThreadWorkerPool
 *  @author Ammar Qammaz (AmmarkoV)
 */

//Can also be compiled using :
//gcc  -O3 example.c -pthread -lm -o example

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include "pthreadWorkerPool.h"

#define NUMBER_OF_WORKER_THREADS 30
#define NUMBER_OF_ITERATIONS 4

struct workerThreadContext
{
    //Add thread specific stuff here..
    double computationInput;
    double computationOutput;
};

void *workerThread(void * arg)
{
    //We are a thread so lets retrieve our variables..
    struct threadContext * ptr = (struct threadContext *) arg;
    fprintf(stdout,"Thread-%u: Started..!\n",ptr->threadID);
    struct workerThreadContext * contextArray = (struct workerThreadContext *) ptr->argumentToPass;
    struct workerThreadContext * ctx = &contextArray[ptr->threadID];
    //----------------------------------------------------

    threadpoolWorkerInitialWait(ptr);
    unsigned int i;
    double work=ctx->computationInput;
    double workStepTwo;

    while (threadpoolWorkerLoopCondition(ptr))
    {
        // This is the location where batch processing work will be carried out.  Right now it is busy-work.
        for ( i = 0; i < 40000000; i++ )
        {
            work = (double) i + 42.23;
            work = sqrt(work);
            workStepTwo = sqrt(work + (double) i);
        }
        ctx->computationOutput = workStepTwo + ptr->threadID;
        //--------------------------------
        threadpoolWorkerLoopEnd(ptr);
    }

    return 0;
}


int main(int argc, char *argv[])
{
    //Our worker pool ready and clean
    struct workerPool pool= {0};

    //We also create one context to be supplied for each thread..
    struct workerThreadContext context[NUMBER_OF_WORKER_THREADS]= {0};

    if ( threadpoolCreate(&pool,NUMBER_OF_WORKER_THREADS,workerThread,(void *) context) )
    {
        fprintf(stdout,"Worker thread pool created.. \n");
        unsigned int iterationID;
        for (iterationID=0; iterationID<NUMBER_OF_ITERATIONS; iterationID++)
        {
            fprintf(stdout,"Iteration %u/%u \n",iterationID+1,NUMBER_OF_ITERATIONS);
            threadpoolMainThreadPrepareWorkForWorkers(&pool);

            fprintf(stdout,"Main thread preparing tasks..!\n");
            //Prepare random input..
            unsigned int contextID;
            for (contextID=0; contextID<NUMBER_OF_WORKER_THREADS; contextID++)
            {
                context[contextID].computationInput = (float)rand()/(float)(RAND_MAX/1000);
            }

            threadpoolMainThreadWaitForWorkersToFinish(&pool);

            fprintf(stdout,"Main thread collecting results..!\n");
            for (contextID=0; contextID<NUMBER_OF_WORKER_THREADS; contextID++)
            {
                fprintf(stdout,"Thread %u / Output : %f\n",contextID,context[contextID].computationOutput);
            }
        }
    }

    fprintf(stdout,"Done with everything..!\n");
    threadpoolDestroy(&pool);
}
