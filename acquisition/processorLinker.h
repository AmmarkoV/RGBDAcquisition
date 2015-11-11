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

   int (*setConfigStr_DisparityMapping) (char * ,char * );
   int (*setConfigInt_DisparityMapping) (char * ,int );

   int (*getDataOutput_DisparityMapping) (unsigned int , unsigned int *  , unsigned int * ,unsigned int * ,unsigned int * );
   int (*addDataInput_DisparityMapping)  (unsigned int , unsigned char * , unsigned int   , unsigned int  ,unsigned int ,unsigned int );

   int (*processData_DisparityMapping) ();

};




/**
 * @brief The array that holds the functions and the state of each plugin
 *        The elements of the array get populated using int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID);
 *        and emptied using int unlinkPlugin(ModuleIdentifier moduleID);
 */
extern struct acquisitionProcessorInterface processors[MAX_NUMBER_OF_PROCESSORS];








int linkToProcessor(char * processorName,char * processorPossiblePath ,char * processorLib ,  int processorID);



#endif // PROCESSORLINKER_H_INCLUDED
