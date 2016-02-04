#include "timer.h"
#include <sys/time.h>
#include <unistd.h>

#include <time.h>

#define EPOCH_YEAR_IN_TM_YEAR 1900

struct TimerArrItem
{
   struct timeval starttime;
   struct timeval endtime;
   struct timeval timediff;

   unsigned int lastTimeMicroseconds;
   unsigned int totalTimeMicroseconds;
   unsigned int timesCounted;
};


unsigned long tickBase = 0;


const char *days[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
/*
int GetDateString(char * output,char * label,unsigned int now,unsigned int dayofweek,unsigned int day,unsigned int month,unsigned int year,unsigned int hour,unsigned int minute,unsigned int second)
{
   //Date: Sat, 29 May 2010 12:31:35 GMT
   //Last-Modified: Sat, 29 May 2010 12:31:35 GMT
   if ( now )
      {
        time_t clock = time(NULL);
        struct tm * ptm = gmtime ( &clock );

        sprintf(output,"%s: %s, %u %s %u %02u:%02u:%02u GMT\n",label,days[ptm->tm_wday],ptm->tm_mday,months[ptm->tm_mon],EPOCH_YEAR_IN_TM_YEAR+ptm->tm_year,ptm->tm_hour,ptm->tm_min,ptm->tm_sec);

      } else
      {
        sprintf(output,"%s: %s, %u %s %u %02u:%02u:%02u GMT\n",label,days[dayofweek],day,months[month],year,hour,minute,second);
      }
    return 1;
}*/

struct TimerArrItem timers_array[TOTAL_TIMERS];


long timeval_diff ( struct timeval *difference, struct timeval *end_time, struct timeval *start_time )
{
  //This returns Microseconds!

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


unsigned int GetLastTimerMilliseconds( unsigned int timer_num )
{
  return (unsigned int) timers_array[timer_num].lastTimeMicroseconds / 1000;
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
 return (float)  1000*1000 / timers_array[timer_num].lastTimeMicroseconds ;
}

void sleepMilliseconds(unsigned int milliseconds)
{
    usleep(milliseconds*1000);
}

void sleepMicroseconds(unsigned int microseconds)
{
    usleep(microseconds);
}

unsigned long GetTickCountMicroseconds()
{
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000000 + ts.tv_nsec/1000;
     return 0;
   }

   return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBase;
}



unsigned long GetTickCountMilliseconds()
{
   //This returns a monotnic "uptime" value in milliseconds , it behaves like windows GetTickCount() but its not the same..
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000 + ts.tv_nsec/1000000;
     return 0;
   }

   return ( ts.tv_sec*1000 + ts.tv_nsec/1000000 ) - tickBase;
}



