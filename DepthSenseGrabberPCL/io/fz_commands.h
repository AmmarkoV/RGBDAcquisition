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

#ifndef __FZ_COMMANDS_HEADER__
#define __FZ_COMMANDS_HEADER__

#include "fz_types.h"

//control operations
#define CMD_DE_SENSOR_START                 0x4001
#define CMD_DE_SENSOR_STOP                  0x4002
#define CMD_DE_RESET                        0x4023
#define CMD_DE_COLOR_SENSOR_START			0x4045
#define CMD_DE_COLOR_SENSOR_STOP			0x4046

//set operation parameters
#define CMD_DE_SET_SHUTTER                  0x4003
#define CMD_DE_SET_FPS                      0x4005
#define CMD_DE_SET_MODE                     0x4008
#define CMD_DE_SET_EXPOSURE                 0x400A //SET_SHUTTER, SET_CMR, SET_FPS combined (CMR is only on device type Jaguar)
#define CMD_DE_SET_FPS_DIVISOR              0x402A

//multishutter operation parameters
#define CMD_DE_SET_SHUTTER_EXT              0x4037 //device type Primesense: one shutter used only
#define CMD_DE_GET_SHUTTER_EXT              0x4036 //device type Primesense: one shutter used only
#define CMD_DE_SET_MS_SATURATION            0x4030 //only device type Jaguar
#define CMD_DE_GET_MS_SATURATION            0x4031 //only device type Jaguar

#pragma pack(2)
typedef struct
{
	uint16_t shutter_nr;   //1 based index (currently shutter 1 or 2). must always be set
	uint16_t shutter_time; //millisec * 10. must be set for CMD_DE_SET_SHUTTER_EXT, is returned for CMD_DE_GET_SHUTTER_EXT
} SHUTTER; //defined here to be used in FZ_SHUTTER_EXT only
typedef struct
{
	uint16_t num_shutters; //must always be set
	SHUTTER shutters[2];
} FZ_SHUTTER_EXT; //defines format of SHUTTER_EXT commands
#pragma pack()

//set filter control and parameters
#define CMD_DE_SET_FILTERCONTROL            0x4101 //only device type Jaguar
#define CMD_DE_SET_LOWLIGHT_GT_PASSIVE      0x4102 //only device type Jaguar
#define CMD_DE_SET_LOWLIGHT_GT_ACTIVE       0x4103 //only device type Jaguar
#define CMD_DE_SET_SATURATION_LT_B          0x4104 //only device type Jaguar
#define CMD_DE_SET_SATURATION_LT_A_MINUS_B  0x4105 //only device type Jaguar
#define CMD_DE_SET_SATURATION_GT_A_MINUS_B  0x4106 //only device type Jaguar
#define CMD_DE_SET_EDGE_FILTER              0x4108 //device type Primesense: ignored
#define CMD_DE_SET_LERP_FILTER              0x4109 //device type Primesense: ignored

//get operation parameters
#define CMD_DE_GET_SHUTTER                  0x5001
#define CMD_DE_GET_FPS                      0x5003
#define CMD_DE_GET_MODE                     0x500C
#define CMD_DE_GET_WOI_TOP                  0x500D //depricated
#define CMD_DE_GET_WOI_BOTTOM               0x500E //depricated
#define CMD_DE_GET_WOI_LEFT                 0x500F //depricated
#define CMD_DE_GET_WOI_RIGHT                0x5010 //depricated
#define CMD_DE_GET_WOI                      0x501B //returns max W,H. depricated, use frame header info instead
#define CMD_DE_GET_FPS_DIVISOR              0x502A
#define CMD_DE_GET_LIGHT_FRQ                0x502B //only device type PA
#define CMD_DE_GET_LIGHT_PWR                0x502C //only device type PA

//get version/serial number information
#define CMD_DE_GET_UNIT_NO                  0x5007
#define CMD_DE_GET_VERSION                  0x5009
#define CMD_CA_GET_VERSION                  0x5012
#define CMD_API_GET_VERSION                 0x5011
#define CMD_DE_GET_PCODE                    0x5018
#define CMD_DE_GET_DEVICE_TYPE              0x5021 //device type Jaguar: fails
#define CMD_DE_GET_CPU_TEMP                 0x5050 //only ARM devices
#define CMD_DE_GET_LED_TEMP                 0x5051 //only ARM devices

//get filter control and parameters
#define CMD_DE_GET_FILTERCONTROL            0x5101 //only device type Jaguar
#define CMD_DE_GET_LOWLIGHT_GT_PASSIVE      0x5102 //only device type Jaguar
#define CMD_DE_GET_LOWLIGHT_GT_ACTIVE       0x5103 //only device type Jaguar
#define CMD_DE_GET_SATURATION_LT_B          0x5104 //only device type Jaguar
#define CMD_DE_GET_SATURATION_LT_A_MINUS_B  0x5105 //only device type Jaguar
#define CMD_DE_GET_SATURATION_GT_A_MINUS_B  0x5106 //only device type Jaguar
#define CMD_DE_GET_EDGE_FILTER              0x5108 //device type Primesense: fails

// Dual sensor commands
#define CMD_DE_REGISTRATION_MODE            0x8030 // 1 = Register to RGB-camera, 0 = off
#define CMD_CA_GET_LDC_CALIB                0x8031 // Internal use
#define CMD_DE_SET_COLOR_TEMP               0x8032 // 2800 - 8000 (K)
#define CMD_DE_AUTO_WHITEBALANCE            0x8033 // 1 = on, 0 = off
#define CMD_DE_AUTO_BLACKLEVEL              0x8034 // 1 = on, 0 = off
#define CMD_DE_SET_SUBSAMPLING              0x8035 // 1 (default), 2, or 4
//--RESPONSE CODES--
#define R_CMD_DE_ACK                        0x6001
#define R_CMD_DE_NACK                       0x6002

//--CMD PARAMETERS--

//CMD_DE_SET_MODE parameter
//jaguar
#define DE_MODE_TEMPORAL                    0x0
#define DE_MODE_SPATIO_TEMPORAL             0x3
#define DE_MODE_MULTI_ST                    0x5
#define DE_MODE_ZFAST                       0x6
#define DE_MODE_ZFINE                       0x7
#define DE_MODE_MS_ST                       0xA
#define DE_MODE_BM_ZFINE                    0xB
#define DE_MODE_BM_TEMPORAL                 0xC
#define DE_MODE_BM_SPATIO_TEMPORAL          0xD
#define DE_MODE_BM_MULTI_ST                 0xE
#define DE_MODE_BM_MS_ST                    0xF
#define DE_MODE_BM_MS_ZFINE                 0x9
#define DE_MODE_BM_MS_ZFINE2                0x8
//panasonic
#define DE_MODE_PA_RAW                      0x10
#define DE_MODE_PA_Z                        0x11
#define DE_MODE_PA_RAW_320                  0x12
#define DE_MODE_PA_Z_MS                     0x1A
//primesense
#define DE_MODE_320X240_30_RAW              0x14
#define DE_MODE_320X240_60_RAW              0x15
#define DE_MODE_640X480_30_RAW              0x16
#define DE_MODE_320X240_30                  0x17
#define DE_MODE_320X240_60                  0x18
#define DE_MODE_640X480_30                  0x19
//primesense_c
#define DE_MODE_320X240_320X240				0x20
#define DE_MODE_320X240_640X480				0x21
#define DE_MODE_640X480_320X240				0x22
#define DE_MODE_640X480_640X480				0x23
#define DE_MODE_320X240_1280X960			0x24
#define DE_MODE_640X480_1280X960			0x25

#endif
