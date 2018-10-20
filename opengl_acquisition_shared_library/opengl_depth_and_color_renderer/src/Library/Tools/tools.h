/** @file tools.h
 *  @brief  Some pretty basic tools for OGL rendering
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED



unsigned long GetTickCountMilliseconds();

/**
* @brief check for the last opengl error occured
* @ingroup OGLTools
* @param String describing the file where the checkOpenGLError is called ( __FILE__ )
* @param Integer describing the line where the checkOpenGLError is called ( __LINE__ )
* @retval 1=Error , 0=No Error
*/
int checkOpenGLError(char * file , int  line);

/**
* @brief Load a file to memory buffer ( returned by this call )
* @ingroup OGLTools
* @param String of filename of the file to load
* @param Output Integer that will say how big the memory chunk loaded is
* @retval 0=Error , A pointer to the block of memory with contents from filename
*/
char * loadFileToMem(char * filename,unsigned long * file_length);

/**
* @brief Cast an unsigned char to a float to represent intensity of a color channel 0-255 to 0-1.0f
* @ingroup OGLTools
* @param Color in unsigned int 0-255
* @retval A float describing the color intensity
*/
float RGB2OGL(unsigned int colr);



#endif // TOOLS_H_INCLUDED
