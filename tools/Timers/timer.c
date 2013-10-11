#include "timer.h"
#include <sys/time.h>
#include <unistd.h>


struct TimerArrItem
{
   struct timeval starttime;
   struct timeval endtime;
   struct timeval timediff;

   unsigned int lastTimeMicroseconds;
   unsigned int totalTimeMicroseconds;
   unsigned int timesCounted;
};


struct TimerArrItem timers_array[TOTAL_TIMERS];


long timeval_diff ( struct timeval *difference, struct timeval *end_time, struct timeval *start_time )
{

   struct timeval temp_diff;

   if(difference==0) { difference=&temp_diff; }

  difference->tv_sec =end_time->tv_sec -start_time->tv_sec ;
  difference->tv_usec=end_time->tv_usec-start_time->tv_usec;

  /* Using while instead of if below makes the code slightly more robust. */

  while(difference->tv_usec<0)
  {
    difference->tv_usec+=1000000;
    difference->tv_sec -=1;
  }

  return 1000000LL*difference->tv_sec+ difference->tv_usec;

}

void StartTimer( unsigned int timer_num )
{
  gettimeofday(&timers_array[timer_num].starttime,0x0);
}

unsigned int EndTimer( unsigned int timer_num )
{
  gettimeofday(&timers_array[timer_num].endtime,0x0);


  timers_array[timer_num].lastTimeMicroseconds = timeval_diff(&timers_array[timer_num].timediff,&timers_array[timer_num].endtime,&timers_array[timer_num].starttime);

  timers_array[timer_num].totalTimeMicroseconds+=timers_array[timer_num].lastTimeMicroseconds;
  ++timers_array[timer_num].timesCounted;


  if ( timers_array[timer_num].totalTimeMicroseconds > 9000000 )
    {
          timers_array[timer_num].totalTimeMicroseconds = timers_array[timer_num].totalTimeMicroseconds / 2;
          timers_array[timer_num].timesCounted = timers_array[timer_num].timesCounted / 2;
    }


  return timers_array[timer_num].lastTimeMicroseconds;
}

unsigned int GetLastTimer( unsigned int timer_num )
{
  return timers_array[timer_num].lastTimeMicroseconds;
}

unsigned int GetAverageTimer( unsigned int timer_num )
{
  if (timers_array[timer_num].timesCounted == 0 ) { return 0; }
  return (unsigned int) timers_array[timer_num].totalTimeMicroseconds/timers_array[timer_num].timesCounted;
}

unsigned int GetTimesTimerTimed( unsigned int timer_num )
{
  return  timers_array[timer_num].timesCounted;
}

float GetFPSTimer( unsigned int timer_num )
{
 if (timers_array[timer_num].lastTimeMicroseconds  == 0 ) { return 0.0; }
 return (float)  1000000 / timers_array[timer_num].lastTimeMicroseconds ;
}

void VisCortxMillisecondsSleep(unsigned int milliseconds)
{
    usleep(milliseconds*1000);
}

void VisCortxMicrosecondsSleep(unsigned int microseconds)
{
    usleep(microseconds);
}
