/*
 * Copyright 2012 Fotonic
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __FZ_API_HEADER__
#define __FZ_API_HEADER__

// definitions for DLL exporting/importing
#if defined(WIN32) && !defined(FZ_API_STATIC)
 #ifdef FZAPI_EXPORTS
  #define FZ_API __declspec(dllexport)
 #else
  #define FZ_API __declspec(dllimport)
 #endif
#else
 #define FZ_API 
#endif

#include <stdio.h>
#include "fz_types.h"
#include "fz_commands.h"

#define FZ_DEFAULT_COMMAND_TCPPORT 1289 // default port on camera for commands (camera replies the port on enumeration)
#define FZ_DEFAULT_IMAGE_TCPPORT   (FZ_DEFAULT_COMMAND_TCPPORT+1) // default port on camera for images

#define FZ_MAX_ROWS_PER_FRAME      1280
#define FZ_MAX_COLS_PER_FRAME      960
#define FZ_MAX_BYTES_PER_PIXEL     14

typedef enum {
	FZ_Success = 0,
	FZ_Failure = 0x1000, // external API errors
	FZ_BAD_HANDLE,
	FZ_BAD_PARAMETERS,
	FZ_INCORRECT_VERSION,
	FZ_INCORRECT_CALIBCONFIG,
	FZ_DEVICE_NOT_FOUND,
	FZ_DEVICE_BUSY,
	FZ_NOT_SUPPORTED,
	FZ_TOO_MANY_DEVICES,
	FZ_STREAM_NOT_RUNNING,
	FZ_STREAM_NO_IMAGES,
	FZ_TIMEOUT,
	FZ_CMD_SEND_FAILED,
	FZ_CMD_RECV_FAILED,
	FZ_NOT_INITIALIZED,
	FZ_NO_CONFIG_FILE_FOUND = 0x3000, // internal API errors
	FZ_NO_CALIB_FILE_FOUND,
	FZ_BAD_CALIB_FILE,
} FZ_Result;

typedef uint32_t FZ_Device_Handle_t;
typedef uint16_t FZ_CmdCode_t;
typedef uint16_t FZ_CmdRespCode_t;

//currently used pixel formats
#define FZ_PIXELFORMAT_B16     0	// 16-bit greyscale in B components
#define FZ_PIXELFORMAT_YUV422  1	// YUV422 in B components

typedef struct
{
	uint8_t u;
	uint8_t y1;
	uint8_t v;
	uint8_t y2;
} FZ_YUV422_DOUBLE_PIXEL;

typedef struct
{
	uint8_t u_or_v;
	uint8_t y;
} FZ_YUV422_DOUBLE_PIXEL_INTERLEAVED;

#pragma pack(4)
typedef struct
{
	uint16_t version;                              // frame version
	uint16_t bytesperpixel;                        // number of bytes in a pixel
	uint16_t nrows;                                // image height
	uint16_t ncols;                                // image width
	uint32_t framecounter;                         // frame number
	uint32_t lasterrorframe;                       // last framecounter with an error
	uint32_t shutter;                              // shutter time (10*ms)
	uint32_t mode;                                 // camera operation mode
	uint32_t reportedframerate;                    // camera frame rate
	uint32_t measuredframerate;                    // frame rate messured by pc
	int32_t  temperature;                          // chip temperature (10*celcius)
	uint32_t timestamp[3];                         // frame time info, see doc
	uint16_t precision_b;                          // max B value (2650 or 1023)
	uint16_t format;                               // data format see FZ_FMT_ below
} FZ_FRAME_HEADER;

#define FZ_DEVICE_TYPE_JAGUAR          0x0
#define FZ_DEVICE_TYPE_PANASONIC       0x1
#define FZ_DEVICE_TYPE_PRIMESENSE      0x2
#define FZ_DEVICE_TYPE_RGBZ            0x3
#define FZ_DEVICE_TYPE_PRIMESENSE_C	   0x4
#define FZ_DEVICE_TYPE_PRIMESENSE_N	   0x5

typedef struct
{
	uint32_t iDeviceType;                          // see above defines
	char     szPath[512];                          // device path to use in FZ_Open
	char     szShortName[32];                      // a more user friendly name
	char     szSerial[16];                         // device serial number
	uint32_t iReserved[64];
} FZ_DEVICE_INFO;
#pragma pack()

#define FZ_FLAG_NO_CFG_CALIB           0x00000002
#define FZ_FLAG_NO_VERSION_CHECK       0x00000004

FZ_API FZ_Result FZ_Init();
FZ_API FZ_Result FZ_Exit();

//DEPRICATED! use FZ_EnumDevices2
FZ_API FZ_Result FZ_GetSimpleDeviceName(
	char* szDevicePath,
	char* szShortName,
	int iShortNameLen);

//DEPRICATED! use FZ_EnumDevices2
FZ_API FZ_Result FZ_EnumDevices(
	char** pszDevicePaths,
	int iMaxDevicePathLen,
	int* piNumDevices);

FZ_API FZ_Result FZ_EnumDevices2(
	FZ_DEVICE_INFO* pDeviceInfo,
	int* piNumDevices);

FZ_API FZ_Result FZ_Open(
	const char* szDevicePath,
	unsigned int iFlags,
	FZ_Device_Handle_t* phDev);

FZ_API FZ_Result FZ_Close(
	FZ_Device_Handle_t hDev);

FZ_API FZ_Result FZ_IOCtl(
	FZ_Device_Handle_t hDev,
	FZ_CmdCode_t iCmd,
	void *pParam,
	int iCmdByteLen,
	FZ_CmdRespCode_t* piRespCode,
	void *pResp,
	int  *piRespByteLen);

#define FZ_FMT_COMPONENT_B             0x0001 //1 short per pixel, order 1 if selected (camera pixel format B16 only)
#define FZ_FMT_COMPONENT_YUV422        0x0002 //2 uint8 per pixel (v,y then u,y), order 1 if selected (camera pixel format YUV422 only)
#define FZ_FMT_COMPONENT_Z             0x0004 //1 short, order 2 if selected
#define FZ_FMT_COMPONENT_XY            0x0008 //2 short, order 3 if selected
#define FZ_FMT_COMPONENT_RADIALZ       0x0010 //1 short, order 4 if selected
#define FZ_FMT_COMPONENT_RGB           0x0020 //4 uint8 (255,b,g,r), order 5 if selected
#define FZ_FMT_PIXEL_INTERLEAVED       0x0100 //components are grouped per pixel ([BZ...][BZ...][BZ...]...)
#define FZ_FMT_PIXEL_PER_PLANE         0x0200 //components are grouped per plane ([WIDTHxB][WIDTHxZ]...)
#define FZ_FMT_PROCESS_MIRROR          0x0400 //mirrors all data positions
#define FZ_FMT_PROCESS_Z_FULLSCALE5M   0x4000 //re-scale Z values to int16 fullscale @ 5m, i.e. (z_mm * 65535) / 5000
#define FZ_FMT_PROCESS_INVERTY         0x8000 //all Y values multiplied with -1

FZ_API FZ_Result FZ_SetFrameDataFmt(
	FZ_Device_Handle_t hDev,
	int x, int y,
	int w, int h,
	int iFlags);

FZ_API FZ_Result FZ_FrameAvailable(
	FZ_Device_Handle_t hDev);

FZ_API FZ_Result FZ_GetFrame(
	FZ_Device_Handle_t hDev,
	FZ_FRAME_HEADER *pHeader,
	void *pPixels,
	size_t *piPixelsByteLen);

FZ_API FZ_Result FZ_GetFrameNewest(
	FZ_Device_Handle_t hDev,
	FZ_FRAME_HEADER *pHeader,
	void *pPixels,
	size_t *piPixelsByteLen);

#define FZ_LOG_NONE             0x0000
#define FZ_LOG_ERROR            0x0001
#define FZ_LOG_WARN             0x0002
#define FZ_LOG_INFO             0x0004
#define FZ_LOG_TRACE            0x0008
#define FZ_LOG_ALL              0xFFFF
#define FZ_LOG_TO_FILE          0x1000 // print to the given file
#define FZ_LOG_TO_STDOUT        0x2000 // print to console window
#define FZ_LOG_OPEN_CONSOLE     0x4000 // use to open a console if program has no console (Windows)

typedef void (*FZ_LOGCALLBACKTYPE)(char *szMetadata, char *szMessage);

FZ_API FZ_Result FZ_SetLogging(
	int iFlags,
	const char *szFilename,
	FZ_LOGCALLBACKTYPE pFunction = NULL);


//frame channel API for sending/receiving images between processes
// it can be used to share an image from one camera to other processes.
//may be useful since a camera can only be opened once.
//this part of the API is totally separate from the other functions.
// it uses a TCP socket at 127.0.0.1:5000+iChannel to communicate.

#define FZ_FRAME_CHAN_SEND 0x1 // opens a channel for sending (connect), blocking with 5sec timeout
#define FZ_FRAME_CHAN_RECV 0x2 // opens a channel for receiving (listen,
// must be active for a sending channel open to succeed), blocking with 10 sec timeout

FZ_API FZ_Result FZ_OpenFrameChannel(
	int iChannel,
	int iFlags);

FZ_API FZ_Result FZ_CloseFrameChannel(
	int iChannel);

FZ_API FZ_Result FZ_SendFrameToChannel(
	int iChannel,
	FZ_FRAME_HEADER *pHeader,
	void *pPixels);

FZ_API FZ_Result FZ_GetFrameFromChannel( //blocking with 5 sec timeout
	int iChannel,
	FZ_FRAME_HEADER *pHeader,
	void *pPixels,
	size_t *piPixelsByteLen);


#ifdef FZAPI_INTERNAL
//note: bOnlyDepthEnginePC is depricated and should be set to false
FZ_API FZ_Result FZ_SetConfigCalib(
	FZ_Device_Handle_t hDev,
	const char *szPath,
	bool bOnlyDepthEnginePC = false);

FZ_API FZ_Result FZ_SetConfigCalib2(
	FZ_Device_Handle_t hDev,
	const char *szPath);

FZ_API FZ_Result FZ_SendTable(
	FZ_Device_Handle_t hDev,
	int iType,
	void *pData,
	int iDataSize);
#endif

//this is the header returned from hardware
#pragma pack(2)
typedef struct
{
	int16_t  magic_number;                         // mark the start of this frame
	uint8_t  version;                              // frame version
	uint8_t  pixelformat;                          // 0 for all but primesense
	uint32_t ExpStartSec;                          // start of exposure in seconds
	uint16_t ExpStartMSec;                         // start of exposure milliseconds
	uint16_t bytesperpixel;                        // number of bytes in a pixel
	uint16_t nrows;                                // image height
	uint16_t ncols;                                // image width
	uint32_t processedframecounter;                // processed frame number
	uint32_t shippedframecounter;                  // shipped frame number
	uint16_t ExpDuration;                          // used as expired time from start of exposure in milliseconds
	int32_t  temperature;                          // chip temperature (10*celcius)
	uint32_t captureid;                            // current frame in the order (useful for raw mode)
	uint32_t shutter;                              // shutter time (10*ms)
	uint32_t cmr;
	uint32_t reportedframerate;                    // camera frame rate
	uint32_t frequency;                            // light pulse frequency
	uint32_t mode;                                 // camera operation mode
	uint32_t measuredframerate;                    // frame rate messured by pc
	uint32_t lasterrorframe;                       // last processedframecounter with an error
	uint16_t dll1;
	uint16_t dll2;
	uint16_t dllerr;
} FZ_FRAME_HEADER_EXT; //defined as FZ_LL_FRAME_HEADER
#pragma pack()

FZ_API FZ_Result FZ_GetFrameExt(
	FZ_Device_Handle_t hDev,
	FZ_FRAME_HEADER_EXT *pHeader,
	void *pPixels,
	size_t *piPixelsByteLen);

FZ_API FZ_Result FZ_GetFrameARGB(
	FZ_Device_Handle_t hDev,
	int *xres, int *yres,
	uint32_t *ExpStartSec,
	uint16_t *ExpStartMSec,
	void *pPixels,
	size_t *piPixelsByteLen);

#endif
