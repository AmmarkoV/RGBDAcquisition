/** @file glx3.h
 *  @brief  X Server bindings to create an OpenGL 3.x+ context and start rendering to a window
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef GLX3_H_INCLUDED
#define GLX3_H_INCLUDED


/**
* @brief create a glx window that can serve OpenGL draw requests
* @ingroup X11
* @param width , The width of the window in pixels
* @param height, The height of the window in pixels
* @param viewWindow, Setting this value to zero will make the "window" invisible
* @param argc, Number of input arguments from main
* @param argv, Pointer to an array of strings from main
* @retval 1=Success , 0=Failure
*/
int start_glx3_stuff(int WIDTH,int HEIGHT,int viewWindow,int argc,const char **argv);


int stop_glx3_stuff();

/**
* @brief After drawing everything on our OpenGL window this call swaps buffers and outputs
* @ingroup X11
* @retval 1=Success , 0=Failure
*/
int glx3_endRedraw();


/**
* @brief Check Window Events , keep window alive
* @ingroup X11
* @retval 1=Success , 0=Failure
*/
int glx3_checkEvents();

#endif // GLX2_H_INCLUDED
