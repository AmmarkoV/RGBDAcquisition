/** @file AmmServerlib.h
* @brief The Main Header for AmmarServer
*
* Any application that may want to interface with AmmarServer will probably want to link to libAmmarServer.a
* and include this header. It provides the entry point for setting up a web share and access to
* sub-modules on runtime.
*
* @author Ammar Qammaz (AmmarkoV)
* @bug AmmarServer is not properly pentested yet
*/

#ifndef AMMSERVERLIB_H_INCLUDED
#define AMMSERVERLIB_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <libgen.h>

/**
* @brief An ID that is used to distinguish between different versions..
* @bug   A potential bug might arise if the specs of the header file are changed and someone is linking with an older version libAmmServer.a thats why this value exists
*/
#define AMMAR_SERVER_HTTP_HEADER_SPEC 136



/**
* @brief An enumerator that lists the types of requests , per HTTP spec , see http://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html
*        Of course not all of them are supported/used internally but they are listed in the same order to maintain spec compatibility
*/
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




/**
* @brief An enumerator that lists the types of requests fields availiable for a POST / GET / COOKIE or FILE request
*/
enum TypesOfRequestFields
{
   NAME        = 1 ,
   VALUE           ,
   FILENAME        ,
   TEMPNAME        ,
   TYPE            ,
   SIZE
};




#define MAX_IP_STRING_SIZE 48 // This should be more than INET6_ADDRSTRLEN
#define MAX_QUERY 2048
#define MAX_RESOURCE 2048
#define MAX_FILE_PATH 1024

/** @brief Size for popen replies */
#define POPEN_BUFFER_SIZE 256

#define MAX_INSTANCE_NAME_STRING 128


/** @brief Maximum number of boundaries to be searched for in a GET request */
#define MAX_HTTP_GET_VARIABLE_COUNT 128

/** @brief Maximum number of boundaries to be searched for in a POST request */
#define MAX_HTTP_POST_BOUNDARY_COUNT 32

/**
* @brief Each POST Request has a payload that consists of a series of boundaries that contain data. This structure holds this data as a series of pointers in the initial header buffer
         in order to facilitate easy and fast parsing of the data
*/
struct POSTRequestBoundaryContent
{
   ///very important to add any fields here to the recalculateHeaderFieldsBasedOnANewBaseAddress  code..
   char * pointerStart;
   char * pointerEnd;
   unsigned int contentSize;

   //All the following are 0 if not used or point to a place inside an HTTP Header
   char *       name;
   unsigned int nameSize;

   char *       value;
   unsigned int valueSize;

   char *       filename;
   unsigned int filenameSize;

   char *       contentDisposition;
   unsigned int contentDispositionSize;

   char *       contentType;
   unsigned int contentTypeSize;

   int reallocateOnHeaderRAWResize;
};


/**
* @brief Each GET Request has a payload that consists of a series of boundaries that contain data. This structure holds this data as a series of pointers in the initial header buffer
         in order to facilitate easy and fast parsing of the data
*/
struct GETRequestContent
{
   ///very important to add any fields here to the recalculateHeaderFieldsBasedOnANewBaseAddress  code..
   //All the following are 0 if not used or point to a place inside an HTTP Header
   char *       name;
   unsigned int nameSize;

   char *       value;
   unsigned int valueSize;

   int reallocateOnHeaderRAWResize;
};


/**
* @brief Each HTTP Request has a header , this is the internal structure that carries the information about the header of an HTTP request parsed and ready for easy
*        for consumption by the various consumers of HTTP requests
*/
struct HTTPHeader
{
   unsigned int failed;
   unsigned int dumpedToFile; //This is dummy

   unsigned int parsingStartOffset;
   unsigned int parsingCurrentLine;


   char * headerRAW; //This is the place where everything rests at..!
   unsigned int headerRAWHeadSize;
   unsigned int headerRAWSize;
   unsigned int headerRAWRequestedSize; // The size that the client requests ( we have our own limits and agenda though )
   unsigned int MAXheaderRAWSize;

   int  requestType; //See enum TypesOfRequests
   char resource[MAX_RESOURCE+1];
   char verified_local_resource[MAX_FILE_PATH+1];

   //char GETquery[MAX_QUERY+1];
   unsigned int sizeOfExtraDataThatWillNeedToBeDeallocated;
   char * extraDataThatWillNeedToBeDeallocated;

   char * GETrequest;
   unsigned long GETrequestSize;

   char * POSTrequest;
   unsigned long POSTrequestSize;
   char * POSTrequestBody;
   unsigned long POSTrequestBodySize;


   unsigned char authorized;
   unsigned char keepalive;
   unsigned char supports_compression;


   //RANGE DATA
   unsigned long range_start;
   unsigned long range_end;


   unsigned long ContentLength; //<- for POST requests

   //The next strings point directly on the header to keep memory usage on a minimum
   //and performance on the maximum , they have to be refreshed if memory gets reallocated:P

   ///very important to add any fields here to the recalculateHeaderFieldsBasedOnANewBaseAddress  code..
   unsigned int cookieLength;
   char * cookie; //<-   *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int hostLength;
   char * host; //<-     *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int refererLength;
   char * referer; //<-  *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int eTagLength;
   char * eTag; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int userAgentLength;
   char * userAgent; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int contentTypeLength;
   char * contentType; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int contentDispositionLength;
   char * contentDisposition;  //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int boundaryLength;
   char * boundary;  //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int GETItemNumber;
   struct GETRequestContent GETItem[MAX_HTTP_GET_VARIABLE_COUNT+1]; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int POSTItemNumber;
   struct POSTRequestBoundaryContent POSTItem[MAX_HTTP_POST_BOUNDARY_COUNT+1]; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int COOKIEItemNumber;
   struct GETRequestContent COOKIEItem[MAX_HTTP_GET_VARIABLE_COUNT+1]; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
};



/**
* @brief Each Dynamic Resource Handler can have multiple profiles for optimizing performance/memory usage etc.
*        For now there are 2 profiles/scenarios. The first one is where there is a global state that all clients should share
*        The second one is where there is a different page for each client , which is more memory intensive since there are separate buffers etc for each request.
*/
enum RHScenarios
{
   SAME_PAGE_FOR_ALL_CLIENTS        = 1 ,
   DIFFERENT_PAGE_FOR_EACH_CLIENT   = 2 ,
   ENABLE_RECEIVING_FILES           = 4
                                  //= 8
                                  //= 16
                                  //= 32
                                  //= 64
};

/**
* @brief We can override/intercept connections before the very fundamental HTTP stage using a request override context and AmmServer_AddRequestHandler
*        This is the structure that holds the information and what to be called back to populate the response
*/
struct AmmServer_RequestOverride_Context
{
   char requestHeader[64]; //Initial request ( GET , HEAD , CONNECT )
   struct HTTPHeader * request;
   void * request_override_callback;
};



/**
* @brief A Wrapper around a memory buffer that enables house keeping for reallocations etc
*/
struct AmmServer_MemoryHandler
{
  unsigned int contentSize;
  unsigned int contentCurrentLength;
  unsigned int contentGrowthStep;
  char * content;
};




/**
* @brief When a call to a function that is a dynamic request is done this is the structure that holds the information
*/
struct AmmServer_DynamicRequest
{
   struct AmmServer_Instance * instance;
   unsigned int clientID;
   unsigned int sessionID;
   unsigned int useSessionLifecycle;

   unsigned int headerResponse;

   char * content;
   unsigned long contentSize;
   unsigned long MAXcontentSize;
   unsigned int contentContainsPathToFileToBeStreamed;

   char * compressedContent;
   unsigned long compressedContentSize;
   unsigned long MAXcompressedContentSize;

   unsigned int sizeOfExtraDataThatWillNeedToBeDeallocated;
   char * extraDataThatWillNeedToBeDeallocated;

   unsigned int POSTItemNumber;
   struct POSTRequestBoundaryContent * POSTItem; //<-    *THIS POINTS SOMEWHERE INSIDE THE STACK SO NEVER FREE IT ..!

   unsigned int GETItemNumber;
   struct GETRequestContent * GETItem; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *

   unsigned int COOKIEItemNumber;
   struct GETRequestContent * COOKIEItem; //<-    *THIS POINTS SOMEWHERE INSIDE headerRAW , or is 0 *
};



/**
* @brief We can override resources to respond with our own C function code , to do so a AmmServer_DynamicRequest must be populated using a AmmServer_AddResourceHandler
*/
struct AmmServer_RH_Context
{
   unsigned int RH_Scenario;

   unsigned int enablePOST;
   unsigned int allowCrossRequests;
   unsigned int needsDifferentPageForEachClient;
   unsigned int needsSamePageForAllClients;
   unsigned int executedNow;
   unsigned int last_callback;
   unsigned int callback_every_x_msec;
   char callback_cooldown;

   void * dynamicRequestCallbackFunction;

   char web_root_path[MAX_FILE_PATH];
   char resource_name[MAX_RESOURCE];

   struct AmmServer_DynamicRequest requestContext;
};

/**
* @brief Each Instance of AmmarServer has some basic settings , which are stored in AmmServer_Instance_Settings
*/
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

    unsigned int MAX_POST_TRANSACTION_SIZE;
    int ENABLE_POST;
};




struct AmmServer_ResponseCategory_Statistics
{
    unsigned long uploadedKB;
    unsigned long downloadedKB;
    unsigned int  requestNumber;
};


/** @brief This holds all the statistics of an Ammar Server Instance
  *        this is the central structure for holding all of these things to keep the rest of the instance from getting cluttered
  * @bug AmmServer_Instance_Statistics are not yet used , they are a stub..!
*/
struct AmmServer_Instance_Statistics
{
    unsigned long filesCurrentlyOpen;
    unsigned long filesTotalOpen;

    unsigned long clientsServed;

    unsigned long totalUploadKB;
    unsigned long totalDownloadKB;


    unsigned long totalUploadBytes;
    unsigned long totalDownloadBytes;


    unsigned long recvOperationsStarted;
    unsigned long recvOperationsFinished;


    struct AmmServer_ResponseCategory_Statistics notModifiedPages;
    struct AmmServer_ResponseCategory_Statistics notFonudPages;
    struct AmmServer_ResponseCategory_Statistics otherCategoriesHere;

    /*etc*/
};


/** @brief This holds all the information about an Ammar Server Instance , sockets , thread pools , cache , memory , settings etc , this is the central structure for holding context */
struct AmmServer_Instance
{
    char instanceName[MAX_INSTANCE_NAME_STRING];
    struct AmmServer_Instance_Settings settings;

    unsigned int prespawn_turn_to_serve;
    unsigned int prespawn_jobs_started;
    unsigned int prespawn_jobs_finished;

    struct AmmServer_Instance_Statistics statistics;

    //Server state
    int serversock;
    int server_running;
    int pause_server;
    int stop_server;

    //Cache Items..
    unsigned int cacheVersionETag; //This should be set to a number ( typically 0 ) and gets prepended to the e-tag , in order to make clients refresh on different launches of AmmarServer
    unsigned long loaded_cache_items_Kbytes;
    unsigned int loaded_cache_items;
    void * cache; /*Actually struct cache_item * but declared as a void pointer here */
    void * cacheHashMap;

    void * clientList;
    void * sessionList;
    //Thread holders..
    unsigned int CLIENT_THREADS_STARTED;
    unsigned int CLIENT_THREADS_STOPPED;

    pthread_t server_thread_id;
    pthread_t * threads_pool;
    //pthread_attr_t attr;

    void * prespawned_pool; //Actually struct PreSpawnedThread * but declared as a void pointer here

    struct AmmServer_RequestOverride_Context * clientRequestHandlerOverrideContext;
    struct AmmServer_RH_Context webserverMonitorPage;
    int webserverMonitorEnabled;

    char webserver_root[MAX_FILE_PATH];
    char templates_root[MAX_FILE_PATH];
};

/** @brief Structure to keep data for an HTTP Transaction */
struct HTTPTransaction
{
  struct AmmServer_Instance * instance;

  struct HTTPHeader incomingHeader;

  //struct HTTPOutHeader outgoingHeader;
  char * outgoingBody;
  unsigned int outgoingBodySize;

  unsigned int resourceCacheID;

  int clientSock;
  int clientDisconnected;
  unsigned int clientListID;
  unsigned int threadID;
  int prespawnedThreadFlag;

  char ipStr[MAX_IP_STRING_SIZE];
  unsigned int port;
};

/** @brief Enumerator for calls AmmServer_GetInfo */
enum AmmServInfos
{
    AMMINF_ACTIVE_CLIENTS=0,
    AMMINF_ACTIVE_THREADS
};

/** @brief Enumerator for calls AmmServer_GetIntSettingValue and AmmServer_SetIntSettingValue */
enum AmmServSettings
{
    AMMSET_PASSWORD_PROTECTION=0,
    AMMSET_RANDOMIZE_ETAG_BEGINNING,
    AMMSET_MAX_POST_TRANSACTION_SIZE,
    AMMSET_TIMEOUT_SEC,
    AMMSET_TEST
};

/** @brief Enumerator for calls AmmServer_GetStrSettingValue and AmmServer_SetStrSettingValue */
enum AmmServStrSettings
{
    AMMSET_USERNAME_STR=0,
    AMMSET_PASSWORD_STR,
    AMMSET_TESTSTR
};

/** @brief Returns a string with the version of AmmarServer , in case it returns NULL it means that we are linked to AmmarServerNULL which means a fake binary */
char * AmmServer_Version();

/**
* @brief Internal Check to compare against changes of the header files
* @ingroup tools
* @param Header ( should be AMMAR_SERVER_HTTP_HEADER_SPEC )
* @retval 1=Success,0=Failure
*/
int AmmServer_CheckIfHeaderBinaryAreTheSame(int headerSpec);




int AmmServer_GetDateString(char * output , unsigned int maxOutput);


int AmmServer_AppendToFile(const char * filename,const char * msg);


/**
* @brief Writes the C string pointed by format to stderr , as a warning ( Yellow ) and logs it to the appropriate log
         If format includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting
         string replacing their respective specifiers.
* @ingroup tools
* @param format , see printf ( http://www.cplusplus.com/reference/cstdio/printf/ )
* @param Arbitrary number of other parameters that where defined in format
*/
void AmmServer_Warning( const char *format , ... );

/**
* @brief Writes the C string pointed by format to stderr , as an error ( Red ) and logs it to the appropriate log
         If format includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting
         string replacing their respective specifiers.
* @ingroup tools
* @param format , see printf ( http://www.cplusplus.com/reference/cstdio/printf/ )
* @param Arbitrary number of other parameters that where defined in format
*/
void AmmServer_Error( const char *format , ... );

/**
* @brief Writes the C string pointed by format to stderr , as a success ( Green ) and logs it to the appropriate log
         If format includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting
         string replacing their respective specifiers.
* @ingroup tools
* @param format , see printf ( http://www.cplusplus.com/reference/cstdio/printf/ )
* @param Arbitrary number of other parameters that where defined in format
*/
void AmmServer_Success( const char *format , ... );



/**
* @brief Writes the C string pointed by format to stderr , as an info ( White ) and does not log it
         If format includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting
         string replacing their respective specifiers.
* @ingroup tools
* @param format , see printf ( http://www.cplusplus.com/reference/cstdio/printf/ )
* @param Arbitrary number of other parameters that where defined in format
*/
void AmmServer_Info( const char *format , ... );



/**
* @brief Writes the C string pointed by format to stderr , as an info ( White ) and does not log it
         If format includes format specifiers (subsequences beginning with %), the additional arguments following format are formatted and inserted in the resulting
         string replacing their respective specifiers.
* @ingroup tools
* @param format , see printf ( http://www.cplusplus.com/reference/cstdio/printf/ )
* @param Arbitrary number of other parameters that where defined in format
*/
void AmmServer_Stub( const char *format , ... );



/**
* @brief Start a Web Server , allocate memory , bind ports and return its instance..
* @ingroup core
* @param String containing the name of this Server
* @param String containing the IP to be binded ( 0.0.0.0 , for all interfaces )
* @param Port to use , ports under 1000 require superuser privileges
* @param String with the filename of a configuration file
* @param String with the root public_html directory , all directories that are childs of this dir could be visible
* @param String with the root directory for templates ( custom 404 pages etc )
* @retval An Ammar Server instance or 0=Failure
*/
struct AmmServer_Instance * AmmServer_Start(const char * name ,const char * ip,unsigned int port,const char * conf_file,const char * web_root_path,const char * templates_root_path);

/**
* @brief Start a Web Server , allocate memory , bind ports and return its instance , also process arguments ( argc and argv from int main(int argc, char *argv[]) ) ..
* @ingroup core
* @param String containing the name of this Server
* @param argc , number of arguments
* @param argv , array of strings
* @param String containing the IP to be binded ( 0.0.0.0 , for all interfaces )
* @param Port to use , ports under 1000 require superuser privileges
* @param String with the filename of a configuration file
* @param String with the root public_html directory , all directories that are childs of this dir could be visible
* @param String with the root directory for templates ( custom 404 pages etc )
* @retval An Ammar Server instance or 0=Failure
*/
struct AmmServer_Instance * AmmServer_StartWithArgs(const char * name , int argc, char ** argv , const char * ip,unsigned int port,const char * conf_file,const char * web_root_path,const char * templates_root_path);

/**
* @brief Stop a Web Server , deallocate memory , free ports and free the server instance..
* @ingroup core
* @param An AmmarServer Instance
* @retval 1=Success,0=Failure
*/
int AmmServer_Stop(struct AmmServer_Instance * instance);

/**
* @brief Query if an instance of AmmarServer is initialized and running
* @ingroup core
* @param An AmmarServer Instance
* @retval 1=Running,0=Stopped
*/
int AmmServer_Running(struct AmmServer_Instance * instance);


/**
* @brief Return a file instead of a Dynamic Request
* @ingroup core
* @param An AmmarServer Request
* @param File to serve
* @retval 1=Running,0=Stopped
*/
int AmmServer_DynamicRequestReturnFile(struct AmmServer_DynamicRequest  * rqst,const char * filename);



void* AmmServer_DynamicRequestReturnMemoryHandler(struct AmmServer_DynamicRequest  * rqst,struct AmmServer_MemoryHandler * content);


/**
* @brief Add a request handler to handle requests , before they get processed internally
*        Calling this will bind a C function that will be called and produce output when someone asks for any resource using the specified method
         TODO : Improve this documenatation
* @ingroup core
* @param An AmmarServer Instance
* @param A AmmServer_RequestOverride_Context to be populated
* @param Request Type
* @param Pointer to function callback
* @retval 1=Success,0=Fail
*/
int AmmServer_AddRequestHandler(struct AmmServer_Instance * instance,struct AmmServer_RequestOverride_Context * RequestOverrideContext,const char * request_type,void * callback);


/**
* @brief Add a scheduler that fires every X seconds  in order to perform mainenance ( defragment databases , save stuff to disk etc )
* @ingroup core
* @param An AmmarServer Instance
* @param Name of resource that will do the callback
* @param Function to be called and provides output when someone asks for resource
* @param Number of seconds between 2 subsequent calls
* @param Number of repetitions to do ( 0 = infinite )
* @retval 1=Success,0=Fail
*/
int AmmServer_AddScheduler
     ( struct AmmServer_Instance * instance,
       const char * resource_name ,
       void * callback,
       unsigned int seconds,
       unsigned int repetitions
    );


/**
* @brief Add a request handler to handle dynamic requests , the core mechanic of AmmarServer
*        Calling this will bind a C function that will be called and produce output when someone asks for a resource
         TODO : Improve this documenatation
* @ingroup core
* @param An AmmarServer Instance
* @param An AmmServer_RH_Context to be populated
* @param Name of resource that should get dynamic responses ( i.e. "index.html" )
* @param Memory chunk to allocate for responses , ( this is the max response size )
* @param Minimum time between two calls of the function ( 0 = no minimum time)
* @param Function to be called and provides output when someone asks for resource
* @param Scenario/Profile of this resource ( see RHScenarios )
* @retval 1=Success,0=Fail
*/
int AmmServer_AddResourceHandler
     ( struct AmmServer_Instance * instance,
       struct AmmServer_RH_Context * context,
       const char * resource_name ,
       //const char * web_root,
       unsigned int allocate_mem_bytes,
       unsigned int callback_every_x_msec,
       void * callback,
       unsigned int scenario
    );


void * AmmServer_EditorCallback(struct AmmServer_DynamicRequest  * rqst);

void * AmmServer_LoginCallback(struct AmmServer_DynamicRequest  * rqst);

//TODO: Add comment here..
int AmmServer_AddEditorResourceHandler
    (
       struct AmmServer_Instance * instance,
       struct AmmServer_RH_Context * context,
       const char * resource_name ,
       //const char * web_root ,
       void * callback
    );

/**
* @brief monitor.html will give information about the server health internals and load , should only be used for debugging
*        otherwise any client will be able to snoop around and see what is happening inside the server
* @ingroup core
* @param An AmmarServer Instance
* @retval 1=Success,0=Fail
*/
int AmmServer_EnableMonitor( struct AmmServer_Instance * instance);


/**
* @brief Remove a request handler that hanles dynamic requests
* @ingroup core
* @param An AmmarServer Instance
* @param An AmmServer_RH_Context to be freed
* @param Switch to control freeing memory or not for this context ( typically should be set to 1 except one knows what he is trying to do )
* @retval 1=Success,0=Failure
*/
int AmmServer_RemoveResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context,unsigned char free_mem);

/**
* @brief Get an Integer out of the state of an instance , of course one can dive into the instance structure but this is a much more clean way to do this
* @ingroup core
* @param An AmmarServer Instance
* @param An ID about which info we want , see (  AmmServInfos )
* @retval Value of the integer we asked about
*/
int AmmServer_GetInfo(struct AmmServer_Instance * instance,unsigned int info_type);

/**
* @brief Get an Integer out of the state of an instance , of course one can dive into the instance structure but this is a much more clean way to do this
* @ingroup core
* @param An AmmarServer Instance
* @param An ID about which integer info we want , see (  AmmServSettings )
* @retval Value of the integer we asked about
*/
int AmmServer_GetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type);

/**
* @brief Set an Integer inside the state of an instance , of course one can dive into the instance structure but this is a much more clean way to do this
* @ingroup core
* @param An AmmarServer Instance
* @param An ID about which integer info we want , see (  AmmServSettings )
* @param New value to set
* @retval Value of the integer we asked about
*/
int AmmServer_SetIntSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,int set_value);

/**
* @brief Get a String out of the state of an instance , of course one can dive into the instance structure but this is a much more clean way to do this
* @ingroup core
* @param An AmmarServer Instance
* @param An ID about which string info we want , see (  AmmServStrSettings )
* @retval Value of the string we asked about
*/
char * AmmServer_GetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type);

/**
* @brief Set an string inside the state of an instance , of course one can dive into the instance structure but this is a much more clean way to do this
* @ingroup core
* @param An AmmarServer Instance
* @param An ID about which integer info we want , see (  AmmServStrSettings )
* @param New string value to set
* @retval 1=Success,0=Failure */
int AmmServer_SetStrSettingValue(struct AmmServer_Instance * instance,unsigned int set_type,const char * set_value);




///-------------------------------------------------------------------------------
/**
* @brief The client can use this call to see how many POST items have been submitted in an HTTP Request
* @ingroup POST
* @param The Dynamic Request we want to examine
* @retval Number of POST items,0=Failure/No Items */
int _POSTnum(struct AmmServer_DynamicRequest * rqst);

/**
* @brief The client can use this call to retrieve a string value of a specific POST item by submitting its name
  The pointer returned is guaranteed to be null terminated although in the case of a large file that can contain additional null characters
  you should use the last variable to know where it properly ends.
* @ingroup POST
* @param The Dynamic Request we want to examine
* @param The name of the POST field we want to receive a string for
* @param Output size of the POST field we selected using our name
* @retval Pointer to our requested value, 0=Failure*/
char * _POST (struct AmmServer_DynamicRequest * rqst,const char * name,unsigned int * valueLength);

/**
* @brief The client can use this call to retrieve the numeric value ( internally delivered using atoi ) of a specific POST items by submitting its name
* @ingroup POST
* @param The Dynamic Request we want to examine
* @param The name of the POST field we want to receive a number for
* @retval Integer value of the POST item with a given name , 0=Failure*/
unsigned int _POSTuint(struct AmmServer_DynamicRequest * rqst,const char * name);

/**
* @brief Quickly check if a POST field has been submitted or not in a request
* @ingroup POST
* @param The Dynamic Request we want to examine
* @param The Name of the POST item we are checking
* @retval 1=Exists , 0=Does not exist*/
int _POSTexists(struct AmmServer_DynamicRequest * rqst,const char * name);

/**
* @brief Shortcut to compare a POST field to a value ( internally using strcmp(POSTValue,ourValue) )
* @ingroup POST
* @param The Dynamic Request we want to examine
* @param The Name of the POST item we are checking
* @param The Value that we want to strncmp with our value
* @retval See strncmp*/
int _POSTcmp(struct AmmServer_DynamicRequest * rqst,const char * name,const char * what2CompareTo);

/**
* @brief Copy a POST field value to a given buffer
* @ingroup POST
* @param The Dynamic Request we want to examine
* @param The Name of the POST item we want to copy checking
* @param The Destination pointer of where we want to copy to
* @param The Destination pointer maximum accommodation size
* @retval 1=Success , 0=Failure*/
int _POSTcpy(struct AmmServer_DynamicRequest * rqst,const char * name,char * destination,unsigned int destinationSize);
///-------------------------------------------------------------------------------



/**
* @brief The client can use this call to see how many GET items have been submitted in an HTTP Request
* @ingroup GET
* @param The Dynamic Request we want to examine
* @retval Number of GET items,0=Failure/No Items */
int _GETnum(struct AmmServer_DynamicRequest * rqst);

/**
* @brief The client can use this call to retrieve a string value of a specific GET item by submitting its name
  The pointer returned is guaranteed to be null terminated although in the case of a large string that can contain additional null characters
  you should use the last variable to know where it properly ends.
* @ingroup GET
* @param The Dynamic Request we want to examine
* @param The name of the GET field we want to receive a string for
* @param Output size of the GET field we selected using our name
* @retval Pointer to our requested value, 0=Failure*/
char * _GET(struct AmmServer_DynamicRequest * rqst,const char * name,unsigned int * valueLength);

/**
* @brief The client can use this call to retrieve the numeric value ( internally delivered using atoi ) of a specific POST items by submitting its name
* @ingroup GET
* @param The Dynamic Request we want to examine
* @param The name of the POST field we want to receive a number for
* @retval Integer value of the POST item with a given name , 0=Failure*/
unsigned int _GETuint(struct AmmServer_DynamicRequest * rqst,const char * name);


/**
* @brief Quickly check if a GET field has been submitted or not in a request
* @ingroup GET
* @param The Dynamic Request we want to examine
* @param The Name of the GET item we are checking
* @retval 1=Exists , 0=Does not exist*/
int _GETexists(struct AmmServer_DynamicRequest * rqst,const char * name);


/**
* @brief Shortcut to compare a GET field to a value ( internally using strcmp(POSTValue,ourValue) )
* @ingroup GET
* @param The Dynamic Request we want to examine
* @param The Name of the GET item we are checking
* @param The Value that we want to strncmp with our value
* @retval See strncmp*/
int _GETcmp(struct AmmServer_DynamicRequest * rqst,const char * name,const char * what2CompareTo);

/**
* @brief Copy a GET field value to a given buffer
* @ingroup GET
* @param The Dynamic Request we want to examine
* @param The Name of the GET item we want to copy checking
* @param The Destination pointer of where we want to copy to
* @param The Destination pointer maximum accommodation size
* @retval 1=Success , 0=Failure*/
int _GETcpy  (struct AmmServer_DynamicRequest * rqst,const char * name,char * destination,unsigned int destinationSize);
///-------------------------------------------------------------------------------
/**
* @brief Shorthand/Shortcut for AmmServer_CookieArg()
* @ingroup shortcut */
const char * _COOKIE(struct AmmServer_DynamicRequest * rqst,const char * name,unsigned int * valueLength);

int _COOKIEnum(struct AmmServer_DynamicRequest * rqst);
///-------------------------------------------------------------------------------
/**
* @brief Shorthand/Shortcut for AmmServer_FILES()
* @ingroup shortcut */
const char * _SESSION(struct AmmServer_DynamicRequest * rqst,const char * name,unsigned int * valueLength);

/**
* @brief Shorthand/Shortcut for AmmServer_FILES()
* @ingroup shortcut */
const char * _FILES(struct AmmServer_DynamicRequest * rqst,const char * POSTName,enum TypesOfRequestFields POSTType,unsigned int * outputSize);
///-------------------------------------------------------------------------------

/**
* @brief Staged way to easily handle bad clients etc from the clients , currently a stub..!
* @ingroup shortcut
* @bug Client behaviours etc are not implemented yet */
int AmmServer_SignalCountAsBadClientBehaviour(struct AmmServer_DynamicRequest * rqst);

/**
* @brief Save Dynamic Request to file
* @ingroup tools
* @param Filename to save the dynamic request
* @param Instance of an AmmarServer
* @param Request that we want to save to a file ( see AmmServer_DynamicRequest )
* @retval 1=Success,0=Failure */
int AmmServer_SaveDynamicRequest(const char* filename  , struct AmmServer_DynamicRequest * rqst);



/**
* @brief Set resource handler to no-cache mode , this means whoever asks for it will never get a cached response
* @ingroup tools
* @param Instance of an AmmarServer
* @param Resource context that should always be served fresh ( AmmServer_RH_Context )
* @retval 1=Success,0=Failure */
int AmmServer_DoNOTCacheResourceHandler(struct AmmServer_Instance * instance,struct AmmServer_RH_Context * context);


/**
* @brief Set resource to no-cache mode , this means whoever asks for it will never get a cached response
* @ingroup tools
* @param Instance of an AmmarServer
* @param Resource name that we want to always serve fresh
* @retval 1=Success,0=Failure */
int AmmServer_DoNOTCacheResource(struct AmmServer_Instance * instance,const char * resource_name);


/**
* @brief Planned functionality for a default http administrator panel per server per instance , currently not implemented correctly
* @ingroup core
* @param IP to bind the interface at
* @param Port to use
* @retval Value of the integer we asked about
*/
struct AmmServer_Instance *  AmmServer_StartAdminInstance(const char * ip,unsigned int port);

/**
* @brief Perform a sanity check on the instance of AmmarServer , this is mostly a dev debug tool and an entry point for code inside AmmServerlib
* @ingroup tools
* @param Ammar Server Instance
* @retval 1=Ok,0=Failed
* @bug Maybe remove AmmServer_SelfCheck
*/
int AmmServer_SelfCheck(struct AmmServer_Instance * instance);



/**
* @brief Execute a command and copy its output line to the provided buffer
* @ingroup tools
* @param Command to execute
* @param Allocated memory to store the result
* @param Size of Allocated memory
* @param Number of line we want to get back
* @retval 1=Ok,0=Failed
* @bug Executing commands can be dangerous , always check and sanitize input before executing , Also be sure about the max size of output so that you don't lose a part of it , also make something like escapeshellcmd
*/
int AmmServer_ExecuteCommandLineNum(const char *  command , char * what2GetBack , unsigned int what2GetBackMaxSize,unsigned int lineNumber);

/**
* @brief Execute a command and copy its output to the provided buffer
* @ingroup tools
* @param Command to execute
* @param Allocated memory to store the result
* @param Size of Allocated memory
* @retval 1=Ok,0=Failed
* @bug Executing commands can be dangerous , always check and sanitize input before executing , Also be sure about the max size of output so that you don't lose a part of it , also make something like escapeshellcmd
*/
int AmmServer_ExecuteCommandLine(const char *  command , char * what2GetBack , unsigned int what2GetBackMaxSize);


/**
* @brief Read a file and store it to a freshly allocated memory block
* @ingroup tools
* @param Input Filename
* @param Output Maximum Size
* @retval Pointer to the new memory or 0=Failed
*/
char * AmmServer_ReadFileToMemory(const char * filename,unsigned int *length );

/**
* @brief Dump a memory block to a file
* @ingroup tools
* @param Output Filename
* @param Input Pointer to memory
* @param Size of memory block
* @retval 1=Ok,0=Failed
*/
int AmmServer_WriteFileFromMemory(const char * filename,const char * memory , unsigned int memoryLength);


/**
* @brief Read a file and store it to a freshly allocated memory handler context
* @ingroup tools
* @param Input Filename
* @retval Pointer to the new memory handler or 0=Failed
*/
struct AmmServer_MemoryHandler *  AmmServer_ReadFileToMemoryHandler(const char * filename);



/**
* @brief Copy a memory handler
* @ingroup tools
* @param Input memory handle
* @retval Pointer to the new memory handler or 0=Failed
*/
struct AmmServer_MemoryHandler *  AmmServer_CopyMemoryHandler(struct AmmServer_MemoryHandler * inpt);


int filterStringForHtmlInjection(char * buffer , unsigned int bufferSize);

/**
* @brief Search for entryPoint pattern in buffer , and inject data there..!
* @ingroup tools
* @param Memory Handler for Buffer we want to inject to , see struct AmmServer_MemoryHandler
* @param String to find in buffer and replace with new content
* @param Data we want to inject
* @retval 1=Ok,0=Failed
*/
int AmmServer_ReplaceVariableInMemoryHandler(struct AmmServer_MemoryHandler * mh,const char * var,const char * value);



/**
* @brief Hot-Replace a character inside a memory block , typically used to replace characters like '+' with ' '
* @ingroup tools
* @param Pointer to memory that contains the null terminated string
* @param Character to be replaced
* @param What to replace the character with
*/
void AmmServer_ReplaceCharInString(char * input , char findChar , char replaceWith);

/**
* @brief Search for entryPoint pattern in buffer multiple times , and inject data in each one..!
* @ingroup tools
* @param Memory Handler for Buffer we want to inject to , see struct AmmServer_MemoryHandler
* @param Number of instances we want to replace
* @param String to find in buffer and replace with new content
* @param Data we want to inject
* @retval 1=Ok,0=Failed
*/
int AmmServer_ReplaceAllVarsInMemoryHandler(struct AmmServer_MemoryHandler * mh ,unsigned int instances,const char * var,const char * value);


int AmmServer_ReplaceAllVarsInDynamicRequest(struct AmmServer_DynamicRequest * dr ,unsigned int instances,const char * var,const char * value);

struct AmmServer_MemoryHandler * AmmServer_AllocateMemoryHandler(unsigned int initialBufferLength, unsigned int growStep);
int AmmServer_FreeMemoryHandler(struct AmmServer_MemoryHandler ** mh);



/**
* @brief Register a function to call a function that gracefully terminates a client when a SIGKILL or the time to stop the server comes
* @ingroup tools
* @param Pointer to function
* @retval 1=Exists,0=Does not Exist
*/
int AmmServer_RegisterTerminationSignal(void * callback);



const char * AmmServer_GetDirectoryFromPath(char * path);
const char * AmmServer_GetBasenameFromPath(char * path);
const char * AmmServer_GetExtensionFromPath(char * path);

/**
* @brief Check if directory Exists
* @ingroup tools
* @param Path to directory
* @retval 1=Exists,0=Does not Exist
*/
int AmmServer_DirectoryExists(const char * filename);


/**
* @brief Check if file Exists
* @ingroup tools
* @param Path to file
* @retval 1=Exists,0=Does not Exist
*/
int AmmServer_FileExists(const char * filename);



int AmmServer_FileIsText(const char * filename);
int AmmServer_FileIsAudio(const char * filename);
int AmmServer_FileIsImage(const char * filename);

/**
* @brief Check if file is a video
* @ingroup tools
* @param Path to file
* @retval 1=Exists,0=Does not Exist
*/
int AmmServer_FileIsVideo(const char * filename);
int AmmServer_FileIsFlash(const char * filename);

/**
* @brief Erase a File
* @ingroup tools
* @param Path to file
* @retval 1=Success,0=Failure
*/
int AmmServer_EraseFile(const char * filename);

/**
* @brief Check if a string has html elements inside it , so if we append it to a web site we won't have html injected
* @ingroup tools
* @param Input String
* @retval 1=Safe,0=Unsafe
*/
unsigned int AmmServer_StringIsHTMLSafe(const char * str);

/**
* @brief Check if a string has html elements inside it , so if we append it to a web site we won't have html injected
* @ingroup tools
* @param Directory path
* @param Input String
* @retval 1=Safe,0=Unsafe
*/
unsigned int AmmServer_StringHasSafePath( const char * directory , const char * filenameUNSANITIZEDString);



#ifdef __cplusplus
}
#endif

#endif // AMMSERVERLIB_H_INCLUDED
