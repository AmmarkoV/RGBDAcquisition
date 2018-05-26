/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef SDK_HELPER_H
#define SDK_HELPER_H

#ifdef WIN32
#pragma warning(disable:4996)
#endif

// includes, project
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <exception.h>
#include <math.h>

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

// includes, timer 

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
 
// namespace unnamed (internal)
namespace 
{
    //! size of PGM file header 
    const unsigned int PGMHeaderSize = 0x40;

    // types

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterFromUByte;

    //! Data converter from unsigned char / unsigned byte 
    template<>
	struct ConverterFromUByte<unsigned char> 
	{
		//! Conversion operator
		//! @return converted value
		//! @param  val  value to convert
		float operator()( const unsigned char& val) 
		{
			return static_cast<unsigned char>(val);
		}
	};

    //! Data converter from unsigned char / unsigned byte to float
    template<>
	struct ConverterFromUByte<float> 
	{
		//! Conversion operator
		//! @return converted value
		//! @param  val  value to convert
		float operator()( const unsigned char& val) 
		{
			return static_cast<float>( val) / 255.0f;
		}
	};

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterToUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<unsigned char> 
    {
        //! Conversion operator (essentially a passthru
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const unsigned char& val) 
        {
            return val;
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<float> 
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()( const float& val) 
        {
            return static_cast<unsigned char>( val * 255.0f);
        }
    };
}

#ifdef _WIN32
	#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
	#define FOPEN_FAIL(result) (result != 0)
	#define SSCANF sscanf_s
#else
	#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
	#define FOPEN_FAIL(result) (result == NULL)
	#define SSCANF sscanf
#endif

inline bool
__loadPPM( const char* file, unsigned char** data, 
         unsigned int *w, unsigned int *h, unsigned int *channels ) 
{
    FILE *fp = NULL;
    if( FOPEN_FAIL(FOPEN(fp, file, "rb")) ) 
    {
        std::cerr << "__LoadPPM() : Failed to open file: " << file << std::endl;
        return false;
    }

    // check header
    char header[PGMHeaderSize];
    if (fgets( header, PGMHeaderSize, fp) == NULL) {
       std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
       return false;
    }
    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else {
        std::cerr << "__LoadPPM() : File is not a PPM or PGM image" << std::endl;
        *channels = 0;
        return false;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    while(i < 3) 
    {
        if (fgets(header, PGMHeaderSize, fp) == NULL) {
            std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
            return false;
        }
        if(header[0] == '#') 
            continue;

        if(i == 0) 
        {
            i += SSCANF( header, "%u %u %u", &width, &height, &maxval);
        }
        else if (i == 1) 
        {
            i += SSCANF( header, "%u %u", &height, &maxval);
        }
        else if (i == 2) 
        {
            i += SSCANF(header, "%u", &maxval);
        }
    }

    // check if given handle for the data is initialized
    if( NULL != *data) 
    {
        if (*w != width || *h != height) 
        {
            std::cerr << "__LoadPPM() : Invalid image dimensions." << std::endl;
        }
    } 
    else 
    {
        *data = (unsigned char*) malloc( sizeof( unsigned char) * width * height * *channels);
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread( *data, sizeof(unsigned char), width * height * *channels, fp) == 0) {
        std::cerr << "__LoadPPM() read data returned error." << std::endl;
    }
    fclose(fp);

    return true;
}

template <class T>
inline bool
sdkLoadPGM( const char* file, T** data, unsigned int *w, unsigned int *h) 
{
    unsigned char* idata = NULL;
    unsigned int channels;
    if( true != __loadPPM(file, &idata, w, h, &channels)) 
    {
        return false;
    }

    unsigned int size = *w * *h * channels;

    // initialize mem if necessary
    // the correct size is checked / set in loadPGMc()
    if( NULL == *data) 
    {
        *data = (T*) malloc( sizeof(T) * size );
    }

    // copy and cast data
    std::transform( idata, idata + size, *data, ConverterFromUByte<T>());

    free( idata );

    return true;
}

template <class T>
inline bool
sdkLoadPPM4( const char* file, T** data, 
               unsigned int *w,unsigned int *h)
{
    unsigned char *idata = 0;
    unsigned int channels;
    
    if (__loadPPM( file, &idata, w, h, &channels)) {
        // pad 4th component
        int size = *w * *h;
        // keep the original pointer
        unsigned char* idata_orig = idata;
        *data = (T*) malloc( sizeof(T) * size * 4);
        unsigned char *ptr = *data;
        for(int i=0; i<size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }
        free( idata_orig);
        return true;
    }
    else
    {
        free ( idata);
        return false;
    }
}

inline bool
__savePPM( const char* file, unsigned char *data, 
         unsigned int w, unsigned int h, unsigned int channels) 
{
    assert( NULL != data);
    assert( w > 0);
    assert( h > 0);

    std::fstream fh( file, std::fstream::out | std::fstream::binary );
    if( fh.bad()) 
    {
        std::cerr << "__savePPM() : Opening file failed." << std::endl;
        return false;
    }

    if (channels == 1)
    {
        fh << "P5\n";
    }
    else if (channels == 3) {
        fh << "P6\n";
    }
    else {
        std::cerr << "__savePPM() : Invalid number of channels." << std::endl;
        return false;
    }

    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) 
    {
        fh << data[i];
    }
    fh.flush();

    if( fh.bad()) 
    {
        std::cerr << "__savePPM() : Writing data failed." << std::endl;
        return false;
    } 
    fh.close();

    return true;
}

template<class T>
inline bool
sdkSavePGM( const char* file, T *data, unsigned int w, unsigned int h) 
{
    unsigned int size = w * h;
    unsigned char* idata = 
      (unsigned char*) malloc( sizeof(unsigned char) * size);

    std::transform( data, data + size, idata, ConverterToUByte<T>());

    // write file
    bool result = __savePPM(file, idata, w, h, 1);

    // cleanup
    free( idata );

    return result;
}

inline bool
sdkSavePPM4ub( const char* file, unsigned char *data, 
              unsigned int w, unsigned int h) 
{
    // strip 4th component
    int size = w * h;
    unsigned char *ndata = (unsigned char*) malloc( sizeof(unsigned char) * size*3);
    unsigned char *ptr = ndata;
    for(int i=0; i<size; i++) {
        *ptr++ = *data++;
        *ptr++ = *data++;
        *ptr++ = *data++;
        data++;
    }

	bool result = __savePPM( file, ndata, w, h, 3);
	free (ndata);
	return result;
}
 
inline bool
sdkLoadPPM4ub( const char* file, unsigned char** data, 
               unsigned int *w, unsigned int *h)
{
    unsigned char *idata = 0;
    unsigned int channels;
    
    if (__loadPPM( file, &idata, w, h, &channels)) {
        // pad 4th component
        int size = *w * *h;
        // keep the original pointer
        unsigned char* idata_orig = idata;
        *data = (unsigned char*) malloc( sizeof(unsigned char) * size * 4);
        unsigned char *ptr = *data;
        for(int i=0; i<size; i++) {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }
        free( idata_orig );
        return true;
    }
    else
    {
        free( idata );
        return false;
    }
}

 

#endif //  SDK_HELPER_H

