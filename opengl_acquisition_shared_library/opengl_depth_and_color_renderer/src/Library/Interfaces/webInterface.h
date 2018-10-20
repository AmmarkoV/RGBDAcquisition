/** @file webInterface.h
 *  @brief  A module that loads models from files so that they can be rendered
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef WEB_INTERFACE_H_INCLUDED
#define WEB_INTERFACE_H_INCLUDED


#include "../TrajectoryParser/TrajectoryParser.h"

int initializeWebInterface(int argc, char *argv[],struct VirtualStream * scene);

#endif


