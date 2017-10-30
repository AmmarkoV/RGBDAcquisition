#ifndef ACQUISITIONSCRIPTINPUT_H_INCLUDED
#define ACQUISITIONSCRIPTINPUT_H_INCLUDED

#include "Acquisition.h"


int getRealModuleAndDevice(
                           struct acquisitionModuleStates * state,
                           ModuleIdentifier * moduleID ,
                           DeviceIdentifier * devID ,
                           unsigned int *width ,
                           unsigned int *height,
                           unsigned int *framerate,
                           char * configuration,
                           char * deviceName,
                           unsigned int stringMaxLength
                          );

int executeScriptFromFile(struct acquisitionModuleStates * state,ModuleIdentifier moduleID,DeviceIdentifier devID,const char * filename);


#endif // ACQUISITIONFILEOUTPUT_H_INCLUDED
