/** @file hashmap.h
* @brief A uniform and clean way to create hashmaps in C and query them
* @author Ammar Qammaz (AmmarkoV)
* @bug This hashmap implementation uses serial searches for now , and needs a lot of work
*/

#ifndef HASHMAP_H_INCLUDED
#define HASHMAP_H_INCLUDED


/** @brief HashMap should always be thread safe , since we are talking about a multi-threaded web-server.
           That being said , if someone wants to use hashmap.c/hashmap.h as a standalone ingredient to another project
           and wants to discard all thread specific locks , it can be easily done with the following switch..!
  */
#define HASHMAP_BE_THREAD_SAFE 1

#if HASHMAP_BE_THREAD_SAFE
   #include <pthread.h>
#else
   #warning "HashMap will not be compiled with thread safe provisions ( pthread )"
#endif // HASHMAP_BE_THREAD_SAFE


/** @brief The function that converts a string to a number so that it will be easier to be searched  */
unsigned long hashFunction(const char *str);

/** @brief An entry on the hash map flattened out for ease of use  */
struct hashMapEntry
{
  unsigned int index;
  unsigned long keyHash;
  unsigned int keyLength;
  char * key;
  unsigned int payloadLength;
  void * payload;
  unsigned int hits;
};

/** @brief The central structure for the hash map */
struct hashMap
{
  unsigned int maxNumberOfEntries;
  unsigned int curNumberOfEntries;
  unsigned int entryAllocationStep;
  struct hashMapEntry * entries;

  void * clearItemCallbackFunction;

  unsigned int isSorted;
  unsigned int useSorting;
  #if HASHMAP_BE_THREAD_SAFE
   pthread_mutex_t hm_addLock;
   pthread_mutex_t hm_fileLock;
  #endif // HASHMAP_BE_THREAD_SAFE
};

/**
* @brief Create and allocate a hash map
* @ingroup hashmap
* @param Number of initial entry space
* @param Allocation step for new allocations
* @param Pointer to a function that clears an item
* @retval Hashmap Structure or , 0=Failure */
struct hashMap * hashMap_Create(unsigned int initialEntries , unsigned int entryAllocationStep,void * clearItemFunction, unsigned int useSorting);

/**
* @brief Destroy and deallocate a hash map
* @ingroup hashmap
* @param HashMap
* @retval 1=Success,0=Failure */
void hashMap_Destroy(struct hashMap * hm);

/**
* @brief Sort hash map
* @ingroup hashmap
* @param HashMap
* @retval 1=Success,0=Failure */
int hashMap_Sort(struct hashMap * hm);


/**
* @brief Hint that we are done adding things to hash map and we are ready for queries
* @ingroup hashmap
* @param HashMap
* @retval 1=Success,0=Failure */
int hashMap_PrepareForQueries(struct hashMap *hm);

/**
* @brief Add a new key to hash map
* @ingroup hashmap
* @param HashMap
* @param String with the key index
* @param String with the value of this record
* @param Length of the value
* @retval 1=Success,0=Failure */
int hashMap_Add(struct hashMap * hm,const char * key,void * val,unsigned int valLength);

/**
* @brief Add a new key ( integer )  to hash map
* @ingroup hashmap
* @param HashMap
* @param String with the key index
* @param Number value of this record
* @retval 1=Success,0=Failure */
int hashMap_AddULong(struct hashMap * hm,const char * key,unsigned long val);

/**
* @brief Find index of a key
* @ingroup hashmap
* @param HashMap
* @param Input String with the key index to find
* @param Output index of the record that holds the data we were searching for
* @retval 1=Success,0=Failure */
int hashMap_FindIndex(struct hashMap * hm,const char * key,unsigned long * index);

/**
* @brief Swap two records
* @ingroup hashmap
* @param HashMap
* @param Index 1 to be swapped
* @param Index 2 to be swapped
* @retval 1=Success,0=Failure */
int hashmap_SwapRecords(struct hashMap * hm , unsigned int index1,unsigned int index2);

/**
* @brief Return key value for index
* @ingroup hashmap
* @param HashMap
* @param Index number
* @retval String of key , or 0 for no key */
char * hashMap_GetKeyAtIndex(struct hashMap * hm,unsigned int index);

/**
* @brief Return key hash for index
* @ingroup hashmap
* @param HashMap
* @param Index number
* @retval Hash of key , or 0 for no key */
unsigned long hashMap_GetHashAtIndex(struct hashMap * hm,unsigned int index);

/**
* @brief Return payload for specified key
* @ingroup hashmap
* @param HashMap
* @param Input String of key
* @param Output Pointer of payload
* @retval 1=Success,0=Failure */
int hashMap_GetPayload(struct hashMap * hm,const char * key,void * payload);

/**
* @brief Return numerical payload for specified key
* @ingroup hashmap
* @param HashMap
* @param Input String of key
* @param Output Pointer of payload
* @retval 1=Success,0=Failure */
int hashMap_GetULongPayload(struct hashMap * hm,const char * key,unsigned long * payload);

/**
* @brief Clear all entries of hash map
* @ingroup hashmap
* @param HashMap
* @retval No return value */
void hashMap_Clear(struct hashMap * hm);

/**
* @brief Check if hashmap contains a key
* @ingroup hashmap
* @param HashMap
* @param String of key
* @retval 1=Exists,0=Does not Exist */
int hashMap_ContainsKey(struct hashMap * hm,const char * key);

/**
* @brief Check if hashmap contains a value
* @ingroup hashmap
* @param HashMap
* @param Value to check for
* @retval 1=Exists,0=Does not Exist */
int hashMap_ContainsValue(struct hashMap * hm,void * val);

/**
* @brief Get the maximum number of entries of hash map
* @ingroup hashmap
* @param HashMap
* @retval Maximum Number of entries */
int hashMap_GetMaxNumberOfEntries(struct hashMap * hm);

/**
* @brief Get the current number of entries of hash map
* @ingroup hashmap
* @param HashMap
* @retval Number of entries */
int hashMap_GetCurrentNumberOfEntries(struct hashMap * hm);




/**
* @brief Console printout of hashmap contents ( useful when debugging )
* @ingroup hashmap
* @param HashMap structure
* @param Title to add
* @retval 1=Success,0=Fail*/
int hashMap_Print(struct hashMap * hm , const char * title);


/**
* @brief Load hash map from a file
* @ingroup hashmap
* @param HashMap structure
* @param Filename to save to
* @retval 1=Success,0=Fail*/
int hashMap_LoadToFile(struct hashMap * hm,const char * filename);

/**
* @brief Save hash map to a file
* @ingroup hashmap
* @param HashMap structure
* @param Filename to save to
* @retval 1=Success,0=Fail*/
int hashMap_SaveToFile(struct hashMap * hm,const char * filename);



/**
* @brief Perform an internal test to check if library is working ok
* @ingroup hashmap
* @retval 1=Success,0=Fail*/
int hashMap_Test(int useSorting);

#endif // HASHMAP_H_INCLUDED
