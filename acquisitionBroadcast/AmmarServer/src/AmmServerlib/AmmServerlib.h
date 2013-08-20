#ifndef AMMSERVERLIB_H_INCLUDED
#define AMMSERVERLIB_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>

enum TypesOfRequests
{
    NONE=0,
    HEAD,
    //Asks for the response identical to the one that would correspond to a GET request, but without the response body. This is useful for retrieving meta-information written in response headers, without having to transport the entire content.
    GET,
    //Requests a representation of the specified resource. Requests using GET should only retrieve data and should have no other effect. (This is also true of some other HTTP methods.)[1] The W3C has published guidance principles on this distinction, saying, "Web application design should be informed by the above principles, but also by the relevant limitations."[11] See safe methods below.
    POST,
    //Submits data to be processed (e.g., from an HTML form) to the identified resource. The data is included in the body of the request. This may result in the creation of a new resource or the updates of existing resources or both.
    PUT,
    //Uploads a representation of the specified resource.
    DELETE,
    //Deletes the specified resource.
    TRACE,
    //Echoes back the received request, so that a client can see what (if any) changes or additions have been made by intermediate servers.
    OPTIONS,
    //Returns the HTTP methods that the server supports for specified URL. This can be used to check the functionality of a web server by requesting '*' instead of a specific resource.
    CONNECT,
    //Converts the request connection to a transparent TCP/IP tunnel, usually to facilitate SSL-encrypted communication (HTTPS) through an unencrypted HTTP proxy.[12]
    PATCH,
    //Is used to apply partial modifications to a resource.[13]
    BAD
};


#define MAX_IP_STRING_SIZE 32
#define MAX_QUERY 512
#define MAX_RESOURCE 512
#define MAX_FILE_PATH 1024
#define MAX_INSTANCE_NAME_STRING 128




struct HTTPHeader
{
   char * headerRAW;
   unsigned int headerRAWSize;

   int  requestType; //See enum TypesOfRequests
   char resource[MAX_RESOURCE+1];
   char verified_local_resource[MAX_FILE_PATH+1];
   char GETquery[MAX_QUERY+1];

   char * POSTrequest;
   unsigned long POSTrequestSize;

   unsigned char authorized;
   unsigned char keepalive;
   unsigned char supports_compression;

   //RANGE DATA
   unsigned long range_start;
   unsigned long range_end;


   unsigned long ContentLength; //<- for POST requests

   //The next strings point directly on the header to keep memory usage on a minimum
   //and performance on the maximum :P
   char * cookie; //<-   *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int cookieLength;

   char * host; //<-     *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int hostLength;

   char * referer; //<-  *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int refererLength;

   char * eTag; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int eTagLength;

   char * userAgent; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int userAgentLength;

   char * contentType; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int contentTypeLength;

   char * contentDisposition;  //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int contentDispositionLength;

   char * boundary;  //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
   unsigned int boundaryLength;

};


struct HTTPOutHeader
{
  unsigned int responseNumber;
};




enum RHScenarios
{
   SAME_PAGE_FOR_ALL_CLIENTS = 0 ,
   DIFFERENT_PAGE_FOR_EACH_CLIENT
};

struct AmmServer_RequestOverride_Context
{
   char requestHeader[64]; //Initial request ( GET , HEAD , CONNECT )
   struct HTTPHeader * request;
   void * request_override_callback;
};



struct AmmServer_DynamicRequest
{
   unsigned int headerResponse;

   char * content;
   unsigned long contentSize;
   unsigned long MAXcontentSize;

   char * compressedContent;
   unsigned long compressedContentSize;
   unsigned long MAXcompressedContentSize;

   char * GET_request;
   unsigned int GET_request_length;

   char * POST_request;
   unsigned int POST_request_length;


   unsigned int clientID;
};


struct AmmServer_RH_Context
{
   unsigned int RH_Scenario;

   unsigned int last_callback;
   unsigned int callback_every_x_msec;
   char callback_cooldown;

   void * dynamicRequestCallbackFunction;

   char web_root_path[MAX_FILE_PATH];
   char resource_name[MAX_RESOURCE];

   struct AmmServer_DynamicRequest requestContext;
};



struct AmmServer_Instance_Settings
{
    //Configuration of instance..
    //------------------------------------------------
    //A ) Password protection..
    int PASSWORD_PROTECTION;
    char * USERNAME;
    char * PASSWORD;
    char * BASE64PASSWORD;
    //------------------------------------------------

    int BINDING_PORT;
};



struct AmmServer_Instance
{
    char instanceName[MAX_INSTANCE_NAME_STRING];
    struct AmmServer_Instance_Settings settings;

    unsigned int prespawn_turn_to_serve;
    unsigned int prespawn_jobs_started;
    unsigned int prespawn_jobs_finished;
    int files_open;

    //Server state
    int serversock;
    int server_running;
    int pause_server;
    int stop_server;

    //Cache Items..
    unsigned long loaded_cache_items_Kbytes;
    unsigned int loaded_cache_items;
    void * cache; /*Actually struct cache_item * but declared as a void pointer here */
    void * cacheHashMap;

    void * clientList;
    //Thread holders..
    unsigned int CLIENT_THREADS_STARTED;
    unsigned int CLIENT_THREADS_STOPPED;

    pthread_t server_thread_id;
    pthread_t * threads_pool;
    //pthread_attr_t attr;

    void * prespawned_pool; //Actually struct PreSpawnedThread * but declared as a void pointer here

    struct AmmServer_RequestOverride_Context * clientRequestHandlerOverrideContext;

    char webserver_root[MAX_FILE_PATH];
    char templates_root[MAX_FILE_PATH];
};




struct HTTPTransaction
{
  struct AmmServer_Instance * instance;

  struct HTTPHeader incomingHeader;

  struct HTTPOutHeader outgoingHeader;
  char * outgoingBody;
  unsigned int outgoingBodySize;

  unsigned int resourceCacheID;

  int clientSock;
  unsigned int clientListID;
  unsigned int threadID;
  int prespawnedThreadFlag;
};


enum AmmServInfos
{
    AMMINF_ACTIVE_CLIENTS=0,
    AMMINF_ACTIVE_THREADS
};


enum AmmServSettings
{
    AMMSET_PASSWORD_PROTECTION=0,
    AMMSET_TEST
};


enum AmmServStrSettings
{
    AMMSET_USERNAME_STR=0,
    AMMSET_PASSWORD_STR,
    AMMSET_TESTSTR
};

char * AmmServer_Version();

void AmmServer_Warning( const char *format , ... );
void AmmServer_Error( const char *format , ... );
void AmmServer_Success( const char *format , ... );

struct AmmServer_Instance * AmmServer_Start(char * name ,char * ip,unsigned int port,char * conf_file,char * web_root_path,char * templates_root_path);

struct AmmServer_Instance * AmmServer_StartWithArgs(char * name , int argc, char ** argv , char * ip,unsigned int port,char * conf_file,char * web_root_path,char * templates_root_path);

int AmmServer_Stop(struct AmmServer_Instance * instance);
int AmmServer_Running(struct AmmServer_Instance * instance);


int AmmServer_AddRequestHandler(struct AmmServer_Instance * instance,struct AmmServer_RequestOverride_Context * RequestOverrideContext,char * request_type,void * callback);

int AmmServer_AddResourceHandler
     ( struct AmmServer_Instance * instance,
       struct AmmServer_RH_Context * context,
       char * resource_name ,
       char * web_root,
       unsigned int allocate_mem_bytes,
       unsigned int callback_every_x_msec,
       void * callback,
       unsigned int scenario
    );

int AmmServer_RemoveResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context,unsigned char free_mem);

int AmmServer_GetInfo(struct AmmServer_Instance * instance,unsigned int info_type);

int AmmServer_GetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type);
int AmmServer_SetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,int set_value);

int AmmServer_POSTArg (struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);
int AmmServer_GETArg  (struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);
int AmmServer_FILES   (struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);

int _POST (struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);
int _GET  (struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);
int _FILES(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst,char * var_id_IN,char * var_value_OUT,unsigned int max_var_value_OUT);

int AmmServer_SignalCountAsBadClientBehaviour(struct AmmServer_Instance * instance,struct AmmServer_DynamicRequest * rqst);

int AmmServer_DoNOTCacheResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context);
int AmmServer_DoNOTCacheResource(struct AmmServer_Instance * instance,char * resource_name);

char * AmmServer_GetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type);
int AmmServer_SetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,char * set_value);

struct AmmServer_Instance *  AmmServer_StartAdminInstance(char * ip,unsigned int port);

int AmmServer_SelfCheck(struct AmmServer_Instance * instance);

int AmmServer_ReplaceVarInMemoryFile(char * page,unsigned int pageLength,char * var,char * value);
char * AmmServer_ReadFileToMemory(char * filename,unsigned int *length );
int AmmServer_WriteFileFromMemory(char * filename,char * memory , unsigned int memoryLength);

int AmmServer_RegisterTerminationSignal(void * callback);

#ifdef __cplusplus
}
#endif

#endif // AMMSERVERLIB_H_INCLUDED
