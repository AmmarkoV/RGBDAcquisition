#ifndef PROCESSORLINKER_H_INCLUDED
#define PROCESSORLINKER_H_INCLUDED

#include "Acquisition.h"

/**
 * @brief This is the structure that contains the calls for each of the plugins loaded
 *
 * Each of this functions is accessed like plugins[moduleID].FUNCTION  initially all of them are 0
 * When the plugin gets loaded they point to the correct address space
 * Connecting them is done with the int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID) and
 * int unlinkPlugin(ModuleIdentifier moduleID);
 * This works great and greatly simplifies the way
 */
struct acquisitionProcessorInterface
{
   void *handle;

   int (*initArgs) (int , char * );

   int (*setConfigStr) (char * ,char * );
   int (*setConfigInt) (char * ,int );

   int (*getDataOutput) (unsigned int , unsigned int *  , unsigned int * ,unsigned int * ,unsigned int * );
   int (*addDataInput)  (unsigned int , void * , unsigned int   , unsigned int  ,unsigned int ,unsigned int );


   unsigned short * (*getDepth) (unsigned int * , unsigned int * ,unsigned int * ,unsigned int * );
   unsigned char * (*getColor)  (unsigned int * , unsigned int * ,unsigned int * ,unsigned int * );


   int (*processData) ();
   int (*cleanup) ();

};




extern unsigned int processorsLoaded;
/**
 * @brief The array that holds the functions and the state of each plugin
 *        The elements of the array get populated using int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID);
 *        and emptied using int unlinkPlugin(ModuleIdentifier moduleID);
 */
extern struct acquisitionProcessorInterface processors[MAX_NUMBER_OF_PROCESSORS];





int bringProcessorOnline(char * processorName,char * processorLibPath,unsigned int *loadedID,int argc, char *argv[]);
int closeAllProcessors();

#endif // PROCESSORLINKER_H_INCLUDED
