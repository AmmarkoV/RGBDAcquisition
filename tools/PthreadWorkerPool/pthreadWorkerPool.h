/** @file pthreadWorkerPool.h
 *  @brief  A header-only thread automization library to make lives easier
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef PTHREADWORKERPOOL_H_INCLUDED
#define PTHREADWORKERPOOL_H_INCLUDED

//The star of the show
#include <pthread.h>

#ifdef __cplusplus
extern "C"
{
#endif


struct threadContext
{
    void * argumentToPass;
    unsigned int threadID;
};


struct workerPool
{
  char initialized;
  char work;
  char mainThreadWaiting;
  //---------------------
  int completedWorkNumber;
  //---------------------
  pthread_attr_t initializationAttribute;
 
  //Start conditions..
  pthread_mutex_t startWorkMutex;
  pthread_cond_t startWorkCondition;
  //---------------------

  //End conditions..
  pthread_mutex_t completeWorkMutex;
  pthread_cond_t completeWorkCondition;
  //---------------------
  
  unsigned int numberOfThreads;
  //---------------------
  struct threadContext *workerPoolContext;
  pthread_t * workerPoolIDs;
};

static char pthreadWorkerPoolVersion[]="0.0";

static int threadpoolCreate(struct workerPool * pool,unsigned int numberOfThreadsToSpawn,void * workerFunction, void * argument)
{
  if (pool==0) { return 0; }
  if (pool->workerPoolIDs!=0) { return 0; }
  if (pool->workerPoolContext!=0) { return 0; }
  
  pool->work = 0;
  pool->mainThreadWaiting = 0;
  pool->numberOfThreads = 0;
  pool->workerPoolIDs     = (pthread_t*) malloc(sizeof(pthread_t) * numberOfThreadsToSpawn);
  pool->workerPoolContext = (struct threadContext*) malloc(sizeof(struct threadContext) * numberOfThreadsToSpawn);

  if (pool->workerPoolIDs==0)     { free(pool->workerPoolContext); pool->workerPoolContext=0; return 0; }
  if (pool->workerPoolContext==0) { free(pool->workerPoolIDs); pool->workerPoolIDs=0;         return 0; }
  
  
  pthread_cond_init(&pool->startWorkCondition,0);
  pthread_mutex_init(&pool->startWorkMutex,0);
  pthread_cond_init(&pool->completeWorkCondition,0);
  pthread_mutex_init(&pool->completeWorkMutex,0);
    
  int threadsCreated = 0;  

  for (unsigned int i=0; i<numberOfThreadsToSpawn; i++)
     {
        pool->workerPoolContext[i].threadID=i;
        pool->workerPoolContext[i].argumentToPass=argument;
        
        int result = pthread_create(
                                    &pool->workerPoolIDs[i],
                                    &pool->initializationAttribute,
                                    workerFunction,
                                    pool->workerPoolContext[i].argumentToPass
                                   ); 
                                   
        threadsCreated += (result == 0);
     }

   pool->numberOfThreads = threadsCreated;
   
   return (threadsCreated==numberOfThreadsToSpawn);
}

static int threadpoolDestroy(struct workerPool *pool)
{
}



#ifdef __cplusplus
}
#endif

#endif // PTHREADWORKERPOOL_H_INCLUDED

