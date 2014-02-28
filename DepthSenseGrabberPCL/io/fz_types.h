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

#ifndef __PROCESSOR_INDEPENDENT_TYPES__
#define __PROCESSOR_INDEPENDENT_TYPES__

#ifdef WIN32 
 #if defined(_MSC_VER) && (_MSC_VER < 1600) //(1600 = Visual Studio 2010)
  typedef int            int32_t;
  typedef unsigned int   uint32_t;
  typedef short          int16_t;
  typedef unsigned short uint16_t;
  typedef char           int8_t;
  typedef unsigned char  uint8_t;
 #else
  #include <stdint.h>
 #endif
#else
 #include <inttypes.h>
#endif

#endif
