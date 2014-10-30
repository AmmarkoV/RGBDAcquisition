/** @file timer.h
 *  @brief This is a small static library to connect RGBDAcquisition to  provide some timer primitives for counting the performance of various
 *  parts of the library and also doing work like counting framerates etc..
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug This does not yet support all operating systems
 */


#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


/**
 * @brief An enumerator to address the different timers availiable for the user
 */
enum VisCortTimerList
{
   UNUSED_TIMER_0 = 0 ,
   UNUSED_TIMER_1 ,
   UNUSED_TIMER_2 ,
   UNUSED_TIMER_3 ,
   UNUSED_TIMER_4 ,
   UNUSED_TIMER_5 ,
   //<- 0-5 reserved for use by clients :P
   FRAME_SNAP_DELAY  ,
   FRAME_PASS_TO_TARGET_DELAY ,
   //----------------------------------------
   TOTAL_TIMERS
};


/**
 * @brief Starts a timer from our pool of timers
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval No Return Values
 */
void StartTimer( unsigned int timer_num );


/**
 * @brief Stops a timer and returns the number of elapsed milliseconds
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval Number of Microseconds
 */
unsigned int EndTimer( unsigned int timer_num );


/**
 * @brief Get the number of elapsed milliseconds from last run of timer
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval Number of Microseconds
 */
unsigned int GetLastTimer( unsigned int timer_num );


/**
 * @brief Get the average number of elapsed milliseconds from all of the runs of the specific timer
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval Number of average Microseconds
 */
unsigned int GetAverageTimer( unsigned int timer_num );

/**
 * @brief Get the count of the times we used the timer
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval Number of uses of the specific timer
 */
unsigned int GetTimesTimerTimed( unsigned int timer_num );

/**
 * @brief Get the framerate of the last run of the timer
 * @ingroup timers
 * @param timerNumber , the number that specifies what timer we want to start
 * @retval FrameRate
 */
float GetFPSTimer( unsigned int timer_num );


/**
 * @brief Sleep for a number of milliseconds
 * @ingroup timers
 * @param milliseconds , to sleep for
 * @retval No Return Value
 */
void sleepMilliseconds(unsigned int milliseconds);


/**
 * @brief Sleep for a number of microseconds
 * @ingroup timers
 * @param microseconds , to sleep for
 * @retval No Return Value
 */
void sleepMicroseconds(unsigned int microseconds);


/**
 * @brief Get elapsed time since the start of RGBDAcquisition ( to set timestamps )
 * @ingroup timers
 * @retval Microseconds since start of RGBDAcquisition
 * @bug GetTickCountMicroseconds segfaults..(?)
 */
unsigned long GetTickCountMicroseconds();


/**
 * @brief Get elapsed time since the start of RGBDAcquisition ( to set timestamps )
 * @ingroup timers
 * @retval Milliseconds since start of RGBDAcquisition
 */
unsigned long GetTickCountMilliseconds();


#ifdef __cplusplus
}
#endif



#endif // VISCORTEXTIMER_H_INCLUDED
