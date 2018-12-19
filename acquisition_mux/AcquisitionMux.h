/** @file AcquisitionMux.h
 *  @brief This is a small library that combines two color/depth frames and fuses them together
 *  @author Ammar Qammaz (AmmarkoV)
 *
 */

#ifndef ACQUISITIONMUX_H_INCLUDED
#define ACQUISITIONMUX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif


/**
 * @brief A dummy enumerator , in the future this will specify different ways to combine 2 streams on to an output stream
 */
enum MuxTypeEnum
{
   COLOR_MUXING = 0,
   DEPTH_MUXING ,
   //----------------------
   UNKNOWN_MUXING
};


/**
 * @brief A Context that internally keeps a record of which modules/devices are beeing combined
 */
struct AcquisitionMuxContext
{
   unsigned int ModuleA;
   unsigned int ModuleB;

   unsigned int DeviceA;
   unsigned int DeviceB;
};


typedef unsigned int AcquisitionMultiplexerIdentifier;



/**
 * @brief Combine A set of base Frames with a set of overlays to produce the muxed output
 * @ingroup mux
 * @param Pointer to the Base RGB Frame input
 * @param Pointer to the Overlay RGB Frame input
 * @param Pointer to the RGB Frame output
 * @param Pointer to the Base Depth Frame input
 * @param Pointer to the Overlay Depth Frame input
 * @param Pointer to the Depth Frame output
 * @param R Channel Value specifying the transparent color ( the overlay background )
 * @param G Channel Value specifying the transparent color ( the overlay background )
 * @param B Channel Value specifying the transparent color ( the overlay background )
 * @param Integer specifying a shift for the overlay in pixels along the X axis
 * @param Integer specifying a shift for the overlay in pixels along the Y axis
 * @param Width  , The Width of the image files ( assumed to be the same for all inputs )
 * @param Height , The Height of the image files ( assumed to be the same for all inputs )
 * @param MuxType , The way to multiplex the base with the overlay , see enum MuxTypeEnum
 * @retval 1 if file Exists , 0 if file does not exist
 * @bug Shifting X/Y does not work for depth stream due ( it is implemented but turned off due to a segfault bug )
 */
int mux2RGBAndDepthFrames(
                           unsigned char * rgbBase, unsigned char * rgbOverlay , unsigned char * rgbOut ,
                           unsigned short * depthBase, unsigned short * depthOverlay , unsigned short * depthOut ,
                           unsigned char transR, unsigned char transG, unsigned char transB ,unsigned char transThreshold,
                           signed int shiftX,signed int shiftY,
                           unsigned int width , unsigned int height , unsigned int rgbTransparency ,
                           unsigned int mux_type
                         );







/**
 * @brief Frame Doubler , Combine A set of frames to produce an intermediate output frame
 * @ingroup mux
 * @param Pointer to the First RGB Frame input
 * @param Pointer to the Second RGB Frame input
 * @param Pointer to the Intermediate RGB Frame output
 * @param Pointer to the First Depth Frame input
 * @param Pointer to the Second Depth Frame input
 * @param Pointer to the Intermediate Depth Frame output
 * @param Width  , The Width of the image files ( assumed to be the same for all inputs )
 * @param Height , The Height of the image files ( assumed to be the same for all inputs )
 * @retval 1 if operation succeeds , 0 if it fails
 */
int generateInterpolatedFrames(
                               unsigned char * firstRGB, unsigned char *  secondRGB , unsigned char * intermediateRGBOut ,
                               unsigned short * firstDepth, unsigned short *  secondDepth  , unsigned short *  intermediateDepthOut  ,
                               unsigned int width ,
                               unsigned int height
                               );


/**
 * @brief  Save a MuxedFrame Buffer to a file
 * @ingroup mux
 * @param String containing the target Filename
 * @param Pointer to the buffer that holds the image we want to store
 * @param Width in pixels
 * @param Height in pixels
 * @param The number of color channels of the buffer , 3 for RGB , 1 for Depth
 * @param The number of bits per pixel 24 for RGB ( 8 bytes for 3 channels ) , 16 for Depth ( 16 bytes for 1 channel )
 * @retval 1 = Success saving the file  , 0 = Failure
 */
int saveMuxImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);

#ifdef __cplusplus
}
#endif


#endif // ACQUISITIONMUX_H_INCLUDED
