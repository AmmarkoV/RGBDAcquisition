/*
AmmarServer , HTTP Server Library

URLs: http://ammar.gr
Written by Ammar Qammaz a.k.a. AmmarkoV 2012

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AmmServerlib.h"

/*
    A Little Rationale here..
    Why on earth does this file exist anyways ? :P

    Some of my projects , i.e. FlashySlideshows depend on AmmarServer so that they can setup a WebInterface..
    In order to (greatly) reduce upkeep cost for all these different projects , and this can be particularly bad when I change the AmmarServer API for example.. :P
    this NullAmmarServer gets included as the "default" webserver ( and it can be compiled without causing a mess ) so the project works out of the box..

    If someone wants the full AmmarServer functionality the ./update_from_git.sh usually clones the real project out of the github repository
*/


char * AmmServer_Version()
{
  return 0;
}


void AmmServer_Warning( const char *format , ... )
{
}

void AmmServer_Error( const char *format , ... )
{
}

void AmmServer_Success( const char *format , ... )
{
}


int AmmServer_Stop(struct AmmServer_Instance * instance)
{
  return 0;
}

struct AmmServer_Instance * AmmServer_Start(char * ip,unsigned int port,char * conf_file,char * web_root_path,char * templates_root_path)
{
  fprintf(stderr,"Binding Null AmmarServer to %s:%u\n",ip,port);

  return 0;
}


int AmmServer_Running(struct AmmServer_Instance * instance)
{
  return 0;
}

int AmmServer_AddRequestHandler(struct AmmServer_Instance * instance,struct AmmServer_RequestOverride_Context * context,char * request_type,void * callback)
{
  return 0;
}


int AmmServer_AddResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context, char * resource_name , char * web_root, unsigned int allocate_mem_bytes,unsigned int callback_every_x_msec,void * callback,unsigned int scenario)
{

  return 0;
}


int AmmServer_PreCacheFile(struct AmmServer_Instance * instance,char * filename)
{
   return 0;
}


int AmmServer_DoNOTCacheResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context)
{
    return 0;
}



int AmmServer_DoNOTCacheResource(struct AmmServer_Instance * instance,char * resource_name)
{
  return 0;
}


int AmmServer_RemoveResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context,unsigned char free_mem)
{
  return 0;
}



int AmmServer_GetInfo(struct AmmServer_Instance * instance,unsigned int info_type)
{
  return 0;
}


int AmmServer_POSTArg(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
  return 0;
}

int AmmServer_GETArg(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
  return 0;
}

int AmmServer_FILES(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
  return 0;
}


int _POST(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
    return 0;
}

int _GET(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
    return 0;
}

int _FILES(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT)
{
    return 0;
}


int AmmServer_GetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type)
{
  return 0;
}

int AmmServer_SetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,int set_value)
{
  return 0;
}


char * AmmServer_GetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type)
{
  return 0;
}

int AmmServer_SetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,char * set_value)
{
  return 0;
}


struct AmmServer_Instance *  AmmServer_StartAdminInstance(char * ip,unsigned int port)
{
  return 0;
}



int AmmServer_SelfCheck(struct AmmServer_Instance * instance)
{
  return 0;
}
int AmmServer_ReplaceVarInMemoryFile(char * page,unsigned int pageLength,char * var,char * value)
{
  return 0;
}

char * AmmServer_ReadFileToMemory(char * filename,unsigned int *length )
{
  return 0;
}

int AmmServer_RegisterTerminationSignal(void * callback)
{
  return 0;
}

